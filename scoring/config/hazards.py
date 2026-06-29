"""Hazard families for the configurable BART.

Each family is a small pydantic model carrying validated numeric parameters and
a single behavioral method, ``hazard_vector(n)``, returning the per-pump
conditional hazard ``h(k) = P(burst at pump k | survived 1..k-1)`` for
``k = 1..n`` (n = the color's ``max_pumps`` cap). Hazards are clamped to [0, 1].

The discriminated union ``HazardSpec`` selects a family from its ``family`` tag,
so a study file can name a family and its parameters without executing code.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


def _clamp(x: float) -> float:
    """Clamp a hazard value into the valid probability range [0, 1]."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class LinearHazard(BaseModel):
    """h(k) = k / N. The default model; burst-time is approximately Rayleigh,
    EV-optimum approximately sqrt(N). Reaches certain burst (h = 1) at the cap."""

    family: Literal["linear"] = "linear"

    def hazard_vector(self, n: int) -> list[float]:
        return [k / n for k in range(1, n + 1)]


class ConstantHazard(BaseModel):
    """h(k) = p (flat). Burst-time is geometric; EV-optimum approximately 1/p."""

    family: Literal["constant"] = "constant"
    p: float = Field(gt=0.0, lt=1.0, description="per-pump burst probability")

    def hazard_vector(self, n: int) -> list[float]:
        return [self.p] * n


class UniformHazard(BaseModel):
    """Classic Lejuez BART: h(k) = 1 / (N - k + 1). Burst point is uniform on
    {1..N}; survival is (N - s)/N, so the EV-optimum sits at N/2."""

    family: Literal["uniform"] = "uniform"

    def hazard_vector(self, n: int) -> list[float]:
        return [1.0 / (n - k + 1) for k in range(1, n + 1)]


class RayleighHazard(BaseModel):
    """h(k) = k / sigma^2. Linear hazard with an explicit scale; EV-optimum is
    approximately sigma. (Equivalent to Linear with N = sigma^2.)"""

    family: Literal["rayleigh"] = "rayleigh"
    sigma: float = Field(gt=0.0, description="Rayleigh scale; optimum ~ sigma")

    def hazard_vector(self, n: int) -> list[float]:
        return [_clamp(k / (self.sigma**2)) for k in range(1, n + 1)]


class ExponentialHazard(BaseModel):
    """h(k) = 1 - e^(-rate) (flat). Geometric burst-time; EV-optimum ~ 1/rate."""

    family: Literal["exponential"] = "exponential"
    rate: float = Field(gt=0.0, description="hazard rate lambda")

    def hazard_vector(self, n: int) -> list[float]:
        h = 1.0 - math.exp(-self.rate)
        return [h] * n


class WeibullHazard(BaseModel):
    """h(k) = (m/N)(k/N)^(m-1) with scale N = cap. Shape m tunes the hazard:
    m<1 decreasing, m=1 flat, m=2 linear-rising, m>2 accelerating."""

    family: Literal["weibull"] = "weibull"
    shape: float = Field(gt=0.0, description="Weibull shape m")

    def hazard_vector(self, n: int) -> list[float]:
        m = self.shape
        return [_clamp((m / n) * (k / n) ** (m - 1)) for k in range(1, n + 1)]


class GompertzHazard(BaseModel):
    """h(k) = a * e^(b*k). Exponentially accelerating hazard (a, b > 0)."""

    family: Literal["gompertz"] = "gompertz"
    a: float = Field(gt=0.0, description="baseline hazard scale")
    b: float = Field(gt=0.0, description="exponential growth rate")

    def hazard_vector(self, n: int) -> list[float]:
        return [_clamp(self.a * math.exp(self.b * k)) for k in range(1, n + 1)]


class LogisticHazard(BaseModel):
    """h(k) = h_max / (1 + e^(-steepness*(k - midpoint))). Safe-then-ramp S-curve."""

    family: Literal["logistic"] = "logistic"
    h_max: float = Field(gt=0.0, le=1.0, description="asymptotic hazard ceiling")
    midpoint: float = Field(gt=0.0, description="pump at which hazard is h_max/2")
    steepness: float = Field(gt=0.0, description="logistic slope r_s")

    def hazard_vector(self, n: int) -> list[float]:
        return [
            _clamp(self.h_max / (1.0 + math.exp(-self.steepness * (k - self.midpoint))))
            for k in range(1, n + 1)
        ]


class LognormalHazard(BaseModel):
    """Hazard of a log-normal burst time: rises then falls (non-monotone)."""

    family: Literal["lognormal"] = "lognormal"
    mu: float = Field(description="log-scale location")
    sigma: float = Field(gt=0.0, description="log-scale shape")

    def hazard_vector(self, n: int) -> list[float]:
        # Hazard h(k) = pdf(k) / sf(k) for a log-normal burst time, computed with
        # the stdlib (no scipy): for a log-normal with parameters (mu, sigma),
        #   sf(k)  = 1/2 * erfc((ln k - mu) / (sigma * sqrt(2)))
        #   pdf(k) = exp(-1/2 * z^2) / (k * sigma * sqrt(2*pi)),  z = (ln k - mu)/sigma
        sqrt2 = math.sqrt(2.0)
        pdf_norm = 1.0 / (self.sigma * math.sqrt(2.0 * math.pi))
        out: list[float] = []
        for k in range(1, n + 1):
            z = (math.log(k) - self.mu) / self.sigma
            sf = 0.5 * math.erfc(z / sqrt2)
            pdf = pdf_norm * math.exp(-0.5 * z * z) / k
            out.append(1.0 if sf <= 1e-12 else _clamp(pdf / sf))
        return out


class StepHazard(BaseModel):
    """Piecewise-constant hazard: ``levels[i]`` applies on segment i, where
    segments are delimited by ascending ``breakpoints`` (len(levels) = len(bp)+1)."""

    family: Literal["step"] = "step"
    breakpoints: list[int] = Field(min_length=1)
    levels: list[float] = Field(min_length=2)

    @model_validator(mode="after")
    def _check_shape(self) -> "StepHazard":
        if len(self.levels) != len(self.breakpoints) + 1:
            raise ValueError("step hazard needs len(levels) == len(breakpoints) + 1")
        if any(self.breakpoints[i] >= self.breakpoints[i + 1] for i in range(len(self.breakpoints) - 1)):
            raise ValueError("step hazard breakpoints must be strictly ascending")
        if self.breakpoints[0] < 1:
            raise ValueError("step hazard breakpoints must be positive pump counts")
        if any(not 0.0 <= lv <= 1.0 for lv in self.levels):
            raise ValueError("step hazard levels must be in [0, 1]")
        return self

    def hazard_vector(self, n: int) -> list[float]:
        out: list[float] = []
        for k in range(1, n + 1):
            seg = 0
            for bp in self.breakpoints:
                if k > bp:
                    seg += 1
                else:
                    break
            out.append(_clamp(self.levels[seg]))
        return out


class TabularHazard(BaseModel):
    """Explicit per-pump hazard array (data, not code). ``values[k-1]`` = h(k)."""

    family: Literal["tabular"] = "tabular"
    values: list[float] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _values_in_range(cls, v: list[float]) -> list[float]:
        if any(not 0.0 <= x <= 1.0 for x in v):
            raise ValueError("tabular hazard values must be in [0, 1]")
        return v

    def hazard_vector(self, n: int) -> list[float]:
        if len(self.values) != n:
            raise ValueError(
                f"tabular hazard has {len(self.values)} values but the color cap is N={n}"
            )
        return [_clamp(x) for x in self.values]


# Discriminated union over ``family``: a study names a family + params, never code.
HazardSpec = Annotated[
    Union[
        LinearHazard,
        ConstantHazard,
        UniformHazard,
        RayleighHazard,
        ExponentialHazard,
        WeibullHazard,
        GompertzHazard,
        LogisticHazard,
        LognormalHazard,
        StepHazard,
        TabularHazard,
    ],
    Field(discriminator="family"),
]
