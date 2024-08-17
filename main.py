import math
import random as rnd
import sympy as sp
from statistics import *

class BinomialDist:
    "Binomial distribution of a random variable"
    # https://en.wikipedia.org/wiki/Binomial_distribution

    __slots__ = {'_n': "n independent experiments", 
                 '_p': "probability of success in each experiment",
                 '_q': "probability of failure in each experiment",
                 '_k': "number of successes in n trials"}
    
    def __init__(self, n: int, p: float, q: float, k: int)-> None:
        """Initializes Binomial Distribution where n is equal to the number of trials and p is the probability of success in each trial 
        and q is the probability of failure in each trial."""
        if n < 0 or k <= 0:
            raise ValueError("n must be greater than 0 and k must be greater than or equal to 0")
        elif p < 0 or p > 1 or q < 0 or q > 1:
            raise ValueError("p must be between 0 and 1 and q must be between 0 and 1")
        elif(q == None or p == None):
            q = 1 - p
            p = 1 - q
        elif (1-p != q):
            raise ValueError("p and q must be complements of each other")
        
        self._n = int(n)
        self._p = float(p)
        self._q = float(q)
        self._k = int(k)

    @classmethod
    def from_samples(cls, data, p: float, k : int) -> "BinomialDist":
        "Make a binomial distribution instance from sample data."
        if(len(data) <= 0 and k < 0):
            raise ValueError("sample data must contain at least one value and k must be greater than or equal to 0")
        if any(x < 0 for x in data):
            raise ValueError("sample data must contain values greater than 0")
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        n = len(data)
        q = 1-p
        return cls(n, p, q, k)
    
    def pmf(self: "BinomialDist") -> float:
        """We write X ~ B(n, p). The probability of getting exactly k successes in n independent Bernoulli trials (with the same rate p) 
        is given by the probability mass function: P(X = k) = C(n, k) * p^k * q^(n-k) where C(n, k) = n! / (k! * (n-k)!) is the binomial coefficient."""
        return math.comb(self._n, self._k) * (self._p ** self._k) * (self._q ** (self._n - self._k))

    def cdf(self: "BinomialDist") -> float:
        """The cumulative distribution function is the probability of getting k or fewer successes in n independent Bernoulli trials (with the same rate p). 
        It is given by the formula: P(X ≤ k) = ∑(i=0, k) C(n, i) * p^i * q^(n-i)"""
        cumulative_prob = 0
        for i in range(self._k):
            self._k = i
            cumulative_prob += self.pmf()
        return 1 - cumulative_prob
    
    @property
    def mean(self: "BinomialDist") -> float:
        return self._n * self._p
    
    @property
    def variance(self: "BinomialDist") -> float:
        return self._n * self._p * self._q
    
    @property
    def std_dev(self: "BinomialDist") -> float:
        return math.sqrt(self.variance)
    
    def __repr__(self: "BinomialDist") -> str:
        return f"{type(self).__name__} with values: (n={self._n}, p={self._p}, q={self._q}, k={self._k})"
    
class HypergeometricDist:
    "Hypergeometric distribution of a random variable"
    # https://en.wikipedia.org/wiki/Hypergeometric_distribution

    __slots__ = {'_N': "total number of objects in the in population", 
                 '_K': "number of success states in the sample",
                 '_n': "total number of trials selected(sample size)",
                 '_k': "number of successes observed(of specified feauture) in n trials"}
    
    def __init__(self, N, n, K, k) -> None:
        """Describes the probability of k successes (random draws for which the object drawn has a specified feature) in n draws, without 
        replacement, from a finite population of size N that contains exactly objects with that feature, wherein each draw is either 
        a success or a failure."""
        if N < 0 or n < 0 or K <= 0:
            raise ValueError("N, n, and K must be greater than or equal to 0")
        elif n > N:
            raise ValueError("n must be less than or equal to ")
        elif k < 0:
            raise ValueError("k must be greater than or equal to 0")
        elif K <= 0:
            raise ValueError("K must be greater than 0")
        
        self._N = int(N)
        self._K = int(K)
        self._n = int(n)
        self._k = int(k)

    @classmethod
    def from_samples(cls, data, n, K, k) -> "HypergeometricDist":
        "Make a hypergeometric distribution instance from sample data."
        if(len(data) <= 0):
            raise ValueError("sample data must contain at least one value and k must be greater than or equal to 0")
        elif k < 0:
            raise ValueError("k must be greater than or equal to 0")
        elif any(x < 0 for x in data):
            raise ValueError("sample data must contain values greater than or euqal to 0")
        N = len(data)
        return cls(N, n, K, k)
    
    def pmf(self: "HypergeometricDist") -> float:
        """We write X ~ H(N, n, r). The probability of getting exactly k successes in n trials (without replacement) is given by the probability mass function: 
        P(X = k) = C(r, k) * C(N-r, n-k) / C(N, n) where C(n, k) = n! / (k! * (n-k)!) is the binomial coefficient."""
        return (math.comb(self._K, self._k) * math.comb(self._N - self._K, self._n - self._k)) / math.comb(self._N, self._n)

    def cdf(self: "HypergeometricDist") -> float:
        """The cumulative distribution function is the probability of getting k or fewer successes in n trials (without replacement). 
        It is given by the formula: P(X ≤ k) = ∑(i=0, k) C(r, i) * C(N-r, n-i) / C(N, n)"""
        cumalative_prob = 0
        for i in range(self._k):
            cumalative_prob += self.pmf()
            self._k-= 1 
        return cumalative_prob
    
    @property
    def mean(self: "HypergeometricDist") -> float:
        """The mean of the hypergeometric distribution is given by the formula: E(X) = n * (K / N)"""
        return self._n * (self._K / self._N)
    
    @property
    def variance(self: "HypergeometricDist") -> float:
        """The variance of the hypergeometric distribution is given by the formula: Var(X) = n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))"""
        return self._n * (self._K / self._N) * ((self._N - self._K) / self._N) * ((self._N - self._n) / (self._N - 1))
    
    @property
    def std_dev(self: "HypergeometricDist") -> float:
        """The standard deviation of the hypergeometric distribution is given by the formula: σ = sqrt(Var(X))"""
        return math.sqrt(self.variance)
    
    def __repr__(self: "HypergeometricDist") -> str:
        return f"{type(self).__name__} with values: (N={self._N}, n={self._n}, K={self._K}, k={self._k})"

class PoissonDist:
    "Poisson distribution of a random variable"
    # https://en.wikipedia.org/wiki/Poisson_distribution

    __slots__ = {'_λ': "average rate of success", 
                 '_k': "number of successes in a fixed interval of time or space"}
    
    def __init__(self, λ, k) -> None:
        """Describes the probability of k successes in a fixed interval of time or space, given that the average rate of success is λ."""
        if λ < 0 or k < 0:
            raise ValueError("λ and k must be greater than or equal to 0")
        
        self._λ = float(λ)
        self._k = int(k)

    @classmethod
    def from_samples(cls, data, k) -> "PoissonDist":
        "Make a poisson distribution instance from sample data."
        if(len(data) <= 0):
            raise ValueError("sample data must contain at least one value and k must be greater than or equal to 0")
        elif k < 0:
            raise ValueError("k must be greater than or equal to 0")
        elif any(x < 0 for x in data):
            raise ValueError("sample data must contain values greater than or euqal to 0")
        λ = mean(data)
        return cls(λ, k)
    
    def pmf(self: "PoissonDist") -> float:
        """We write X ~ P(λ). The probability of getting exactly k successes in a fixed interval of time or space is given by the probability mass function: 
        P(X = k) = (λ^k * e^(-λ)) / k!"""
        return (self._λ ** self._k) * math.exp(-self._λ) / math.factorial(self._k)

    def cdf(self: "PoissonDist") -> float:
        """The cumulative distribution function is the probability of getting k or fewer successes in a fixed interval of time or space. 
        It is given by the formula: P(X ≤ k) = ∑(i=0, k) (λ^i * e^(-λ)) / i!"""
        cumalative_prob = 0
        for i in range(self._k):
            cumalative_prob += self.pmf()
            self._k-= 1 
        return cumalative_prob
    
    @property 
    def mean(self: "PoissonDist") -> float:
        """The mean of the poisson distribution is given by the formula: E(X) = λ"""
        return self._λ
    
    @property
    def variance(self: "PoissonDist") -> float:
        """The variance of the poisson distribution is given by the formula: Var(X) = λ"""
        return self._λ
    
    @property
    def std_dev(self: "PoissonDist") -> float:
        """The standard deviation of the poisson distribution is given by the formula: σ = sqrt(λ)"""
        return math.sqrt(self._λ)
    
    def __repr__(self: "PoissonDist") -> str:
        return f"{type(self).__name__} with values: (λ={self._λ}, k={self._k})"
    
class GeometricDist:
    "Geometric distribution of a random variable"
    # https://en.wikipedia.org/wiki/Geometric_distribution

    __slots__ = {'_p': "probability of success in each trial", 
                 '_q': "probability of failure in each trial",
                 '_k': "number of trials until the first success"}
    
    def __init__(self, p, q, k) -> None:
        """Make an instance of a Geometric Distribution, where the geometric distribution describes
        the number of trials it takes to achieve the first success in a sequence of independent Bernoulli trials."""
        
        if p < 0 or p > 1 or q < 0 or q > 1:
            raise ValueError("p must be between 0 and 1 and q must be between 0 and 1")
        elif(q == None or p == None):
            q = 1 - p
            p = 1 - q
        elif (1-p != q):
            raise ValueError("p and q must be complements of each other")
        elif k < 0:
            raise ValueError("k must be greater than or equal to 0")
        
        self._p = float(p)
        self._q = float(q)
        self._k = int(k)

    @classmethod
    def from_samples(cls, data, k) -> "GeometricDist":
        "Make a geometric distribution instance from sample data."
        if(len(data) <= 0):
            raise ValueError("sample data must contain at least one value and k must be greater than or equal to 0")
        elif k < 0:
            raise ValueError("k must be greater than or equal to 0")
        elif any(x < 0 for x in data):
            raise ValueError("sample data must contain values greater than or euqal to 0")
        p = mean(data)
        q = 1 - p
        return cls(p, q, k)
    
    def pmf(self: "GeometricDist") -> float:
        """We write X ~ G(p). The probability of getting exactly k failures before the first success in a sequence of Bernoulli trials is given by the probability mass function: 
        P(X = k) = q^k * p"""
        return (self._q ** self._k) * self._p

    def cdf(self: "GeometricDist") -> float:
        """The cumulative distribution function is the probability of getting k or fewer failures before the first success in a sequence of Bernoulli trials."""
        cumulative_prob = 0
        for i in range(self._k):
            self._k = i
            cumulative_prob += self.pmf()
        return 1 - cumulative_prob

    @property
    def mean(self: "GeometricDist") -> float:
        """The mean of the geometric distribution is given by the formula: E(X) = 1 / p"""
        return 1 / self._p
    
    @property
    def variance(self: "GeometricDist") -> float:
        """The variance of the geometric distribution is given by the formula: Var(X) = q / p^2"""
        return self._q / (self._p ** 2)
    
    @property
    def std_dev(self: "GeometricDist") -> float:
        """The standard deviation of the geometric distribution is given by the formula: σ = sqrt(q / p^2)"""
        return math.sqrt(self.variance)
    
    def __repr__(self: "GeometricDist") -> str:
        return f"{type(self).__name__} with values: (p={self._p}, q={self._q}, k={self._k})"
    
class UniformDist:
    "Uniform distribution of a random variable"
    # https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    
    __slots__ = {'_a': "minimum value", 
                 '_b': "maximum value",
                 '_x': "random variable",
                 "-_n": "number of equally likely outcomes"}
    
    def __init__(self, a, b, x) -> None:
        """Discrete uniform distribution is a probability distribution that describes the likelihood of outcomes when each outcome in a 
        finite set is equally likely. Make an instance of a Uniform Distribution, where the uniform distribution describes the probability 
        of a random variable taking on a value within a given range."""
        if a < 0 or b < 0 or x < 0:
            raise ValueError("a, b, and x must be greater than or equal to 0")
        elif a > b:
            raise ValueError("a must be less than or equal to b")
        elif x < a or x > b:
            raise ValueError("x must be greater than or equal to a and less than or equal to b")
        self._a = float(a)
        self._b = float(b)
        self._x = float(x)
        self._n = int(b - a)

    @classmethod
    def from_samples(cls, data, x) -> "UniformDist":
        "Make a uniform distribution instance from sample data."
        if(len(data) <= 0):
            raise ValueError("sample data must contain at least one value and x must be greater than or equal to 0")
        elif x < 0:
            raise ValueError("x must be greater than or equal to 0")
        elif any(x < 0 for x in data):
            raise ValueError("sample data must contain values greater than or euqal to 0")
        a = min(data)
        b = max(data)
        return cls(a, b, x)
    
    def pmf(self: "UniformDist") -> float:
        """We write X ~ U(a, b). The probability mass function(a.k.a probability density function) of a uniform distribution is given by the formula: 
        f(x) = 1 / n"""
        return 1 / (self._n)

    def cdf(self: "UniformDist") -> float:
        """The cumulative distribution function is the probability of getting a value less than or equal to x in a uniform distribution.
        f(x) = (x - a + 1) / n"""
        return (self._x - self._a + 1) / (self._n)
    
    @property
    def mean(self: "UniformDist") -> float:
        """The mean of the uniform distribution is given by the formula E(X) = (a + b) / 2"""
        return (self._a + self._b) / 2
    
    @property
    def variance(self: "UniformDist") -> float:
        """The variance of the uniform distribution is given by the formula Var(X) = (n^2 - 1) / 12"""
        return (self._n ** 2 - 1) / 12
    
    @property
    def std_dev(self: "UniformDist") -> float:
        """The standard deviation of the uniform distribution is given by the formula σ = sqrt((n^2 - 1) / 12)"""
        return math.sqrt(self.variance)
    
    def __repr__(self: "UniformDist") -> str:
        return f"{type(self).__name__} with values: (a={self._a}, b={self._b}, x={self._x})"
    
class ExponentialDist:
    "Exponential distribution of a random variable"
    # https://en.wikipedia.org/wiki/Exponential_distribution
    
    __slots__ = {'_λ': "rate of success", 
                 '_x': "random variable"}
    
    def __init__(self, λ, x) -> None:
        """The exponential distribution in R Language is the probability distribution of the time between events in a Poisson point 
        process, i.e., a process in which events occur continuously and independently at a constant average rate.
        Describes the probability of a random variable taking on a value within a given range, given that the rate of success is λ."""
        if λ < 0 or x < 0:
            raise ValueError("λ and x must be greater than or equal to 0")
        self._λ = float(λ)
        self._x = float(x)
    
    @classmethod
    def from_samples(cls, data, x) -> "ExponentialDist":
        "Make an exponential distribution instance from sample data."
        if(len(data) <= 0):
            raise ValueError("sample data must contain at least one value and x must be greater than or equal to 0")
        elif x < 0:
            raise ValueError("x must be greater than or equal to 0")
        elif any(x < 0 for x in data):
            raise ValueError("sample data must contain values greater than or equal to 0")
        λ = 1 / mean(data)
        return cls(λ, x)
    
    def pmf(self: "ExponentialDist") -> float:
        """We write X ~ Exp(λ). The probability of getting a value less than or equal to x in an exponential distribution is given by the formula: 
        f(x) = λ * e^(-λ * x)"""
        return self._λ * math.exp(-self._λ * self._x)
    
    def cdf(self: "ExponentialDist") -> float:
        """The cumulative distribution function is the probability of getting a value less than or equal to x in an exponential distribution.
        F(x) = 1 - e^(-λ * x)"""
        return 1 - math.exp(-self._λ * self._x)
    
    @property
    def mean(self: "ExponentialDist") -> float:
        """The mean of the exponential distribution is given by the formula E(X) = 1 / λ"""
        return 1 / self._λ
    
    @property
    def variance(self: "ExponentialDist") -> float:
        """The variance of the exponential distribution is given by the formula Var(X) = 1 / λ^2"""
        return 1 / (self._λ ** 2)
    
    @property
    def std_dev(self: "ExponentialDist") -> float:
        """The standard deviation of the exponential distribution is given by the formula σ = sqrt(1 / λ^2)"""
        return math.sqrt(self.variance)
    
    def __repr__(self: "ExponentialDist") -> str:
        return f"{type(self).__name__} with values: (λ={self._λ}, x={self._x})"

class NegativeBinomialDist:
    "Negative Binomial distribution of a random variable"
    # https://en.wikipedia.org/wiki/Negative_binomial_distribution

    __slots__ = {'_r': "number of successes", 
                 '_p': "probability of success in each trial",
                 '_q': "probability of failure in each trial",
                 '_k': "number of trials until the rth success"}
    
    def __init__(self, r, p, q, k) -> None:
        """The negative binomial distribution describes the number of trials(failures) it takes to achieve the rth success in a sequence of independent Bernoulli trials.
        Make an instance of a Negative Binomial Distribution, where the negative binomial distribution describes the probability of a random variable taking on a value within a given range."""
        if r < 0 or p < 0 or p > 1 or q < 0 or q > 1 or k < 0:
            raise ValueError("r, p, q, and k must be greater than or equal to 0")
        elif(q == None or p == None):
            q = 1 - p
            p = 1 - q
        elif (1-p != q):
            raise ValueError("p and q must be complements of each other")
        
        self._r = int(r)
        self._p = float(p)
        self._q = float(q)
        self._k = int(k)

    @classmethod
    def from_samples(cls, data, r, k) -> "NegativeBinomialDist":
        "Make a negative binomial distribution instance from sample data."
        if(len(data) <= 0):
            raise ValueError("sample data must contain at least one value and k must be greater than or equal to 0")
        elif k < 0:
            raise ValueError("k must be greater than or equal to 0")
        elif any(x < 0 for x in data):
            raise ValueError("sample data must contain values greater than or euqal to 0")
        p = mean(data)
        q = 1 - p
        return cls(r, p, q, k)
    
    def pmf(self: "NegativeBinomialDist") -> float:
        """We write X ~ NB(r, p). The probability of getting exactly k failures before the rth success in a sequence 
        of Bernoulli trials is given by the probability mass function: P(X = k) = C(k + r - 1, r - 1) * p^r * q^k"""
        return math.comb(self._k + self._r - 1, self._r - 1) * (self._p ** self._r) * (self._q ** self._k)
    
    def cdf(self: "NegativeBinomialDist") -> float:
        """The cumulative distribution function is the probability of getting k or fewer failures before the rth success 
        in a sequence of Bernoulli trials. X~NB(r, p). It is given by the formula: P(X ≤ k) = Ip(r, k+1)"""
        return self.I_p(self._r, self._k + 1, self._p)
    
    @classmethod
    def I_p(r: float, k: float, x: float) -> float:
        """The incomplete regularized beta function is given by the formula: I_p(a, b, x) = 1 - I_(1-p)(k, r, 1-x). 
        Where r is the number of successes, k is the number of trials until the rth success, and x is the probability 
        of success in each trial."""
        return 1 - sp.betainc(k, r, 1 - x)
    
    @property
    def mean(self: "NegativeBinomialDist") -> float:
        """The mean of the negative binomial distribution is given by the formula E(X) = r / p"""
        return self._r / self._p
    
    @property
    def variance(self: "NegativeBinomialDist") -> float:
        """The variance of the negative binomial distribution is given by the formula Var(X) = r * q / p^2"""
        return self._r * self._q / (self._p ** 2)
    
    @property
    def std_dev(self: "NegativeBinomialDist") -> float:
        """The standard deviation of the negative binomial distribution is given by the formula σ = sqrt(r * q / p^2)"""
        return math.sqrt(self.variance)
    
    def __repr__(self: "NegativeBinomialDist") -> str:
        return f"{type(self).__name__} with values: (r={self._r}, p={self._p}, q={self._q}, k={self._k})"
    

def main():
    # # Example 1
    # data = []
    # for i in range(15):
    #     data.append(rnd.randint(10,55))
    # print(50*"~*~")
    # binomial = BinomialDist(15, 0.05, 0.95, 2)
    # print(f"Before using the class method from_samples: {binomial._n}, {binomial._p}, {binomial._k}")
    # print(f"Probability Mass Function: {binomial.pmf():.3f}")
    # print(50*"~*~")
    # print("\n\n")
    # print(50*"~*~")
    # binomial = BinomialDist.from_samples(data, 0.05, 4)
    # # print(f"After using the class method from_samples: {binomial._n}, {binomial._p}, {binomial._k}")
    # # print(f"Probability Mass Function: {binomial.pmf():.3f} ")
    # # print(50*"~*~")

    # print("\n\n")
    # print(50*"~*~")
    # print(f"After using the class method from_samples: {binomial._n}, {binomial._p}, {binomial._k}")
    # print(f"The Cumulative Distribution Function: {1-binomial.cdf():.3f}")
    # print(50*"~*~")


    # # Example 2 of repr
    # print(50*"~*~")
    # print(binomial)
    # print(50*"~*~")
    print(sp.__docs__)
    #Example of Hypergeometric Distribution
    data = [] 
    for i in range(15):
        data.append(rnd.randint(10,55))
    print(50*"~*~")
    hypergeometric = HypergeometricDist(25, 10, 15, 5)
    print(hypergeometric)
    print(f"Before using the class method from_samples: {hypergeometric._N}, {hypergeometric._n}, {hypergeometric._k}")
    print(f"Probability Mass Function: {hypergeometric.pmf():.3f}")
    print(f"Cumulative denisty function: {hypergeometric.cdf():.10f}")
    print(hypergeometric.__repr__())
    print(50*"~*~")
    print("\n\n")
    print(50*"~*~")
    print(data)
    hypergeometric = HypergeometricDist.from_samples(data, 10, 5, 2)
    print(hypergeometric)
    print(f"After using the class method from_samples: {hypergeometric._N}, {hypergeometric._n}, {hypergeometric._k}")
    print(f"Probability Mass Function: {hypergeometric.pmf():.3f}")
    print(f"Cumulative denisty function: {hypergeometric.cdf():.10f}")
    print(hypergeometric.__repr__())
    print(50*"~*~")
    print("\n\n")
    print(50*"~*~")

if __name__ == "__main__":
    main()
