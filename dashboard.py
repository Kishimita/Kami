# Packages
import math
import sympy as sp
from shiny import ui, App, render, reactive
from statistics import mean, stdev, variance

# Discrete
from distributions import BinomialDist
from distributions import PoissonDist
from distributions import GeometricDist
from distributions import HypergeometricDist
from distributions import NegativeBinomialDist

# Continuous
from distributions import UniformDist
from distributions import ExponentialDist
from distributions import StudentDist
from distributions import FDist
from distributions import GammaDist
from distributions import ChiSquaredDist

# --------------------------------------- UI Section --------------------------------------- 
app_ui = ui.page_sidebar(

    ui.sidebar(
        #Creates choices "Discrete" and "Continuous" with ID: distribution_type
        ui.input_select("distribution_type", "Select Type", choices=["Discrete", "Continuous"]),
        ui.output_ui("distribution_choices"),
        title= "Distribution Selector"
    ),

    # outputs
    ui.output_ui("distribution_info"),
    ui.output_ui("distribution_inputs"),
    ui.output_ui("distribution_results"),
    title="Probability Distribution Calculator"

)

#--------------------------------------- Server Section ------------------------------------- 
def server(input, output, session):

    # Distribution Choices
    @output
    @render.ui
    def distribution_choices():
        """
        This method creates the distribution options from the UI section. 

        Note: distribution_type ID is contained within app_ui
        """

        # If distribution_type links to discrete, returns discrete distributions
        if input.distribution_type() == "Discrete":
            return ui.input_radio_buttons("specific_distribution", "Select Distribution", 
                                          choices=["Binomial", "Poisson", "Geometric", "Hypergeometric", "Negative Binomial"])
        
        # If distribution_type links to Continuous, returns continuous distributions 
        elif input.distribution_type() == "Continuous":
            return ui.input_radio_buttons("specific_distribution", "Select Distribution", 
                                          choices=["Uniform", "Exponential", "Student", "F", "Gamma", "Chi-square"])
        
        else:
            return ui.p("Select a distribution type")


    # Distribution information
    @output
    @render.ui
    def distribution_info():
        """
        Depending on the distribution selected from distribution choices method, it displays the
        information corresponding with the chosen distribution

        Note: specific_distribution ID is contained within distribution_choices 
        """
        
        selected = input.specific_distribution() # line 23 and 26 contain ID

        # ------------------------------ Discrete Distributions ------------------------------------

        # Binomial Section
        if selected == "Binomial":
            return ui.div(
                ui.h2("Binomial Distribution"),
                ui.p("The binomial distribution models the number of successes in a fixed number of independent Bernoulli trials."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("n: number of trials"),
                    ui.tags.li("p: probability of success on each trial")
                )
            )
        
        # Poisson Section
        elif selected == "Poisson":
            return ui.div(
                ui.h2("Poisson Distribution"), 
                ui.p("The Poisson distribution models the number of events occurring in a fixed interval of time or space."), 
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("λ (lambda): average number of events per interval"),
                    ui.tags.li("k: number of successes in a fixed interval of time or space")
                )
            )
        
        # Geometric Section
        elif selected == "Geometric":
            return ui.div(
                ui.h2("Geometric Distribution"),
                ui.p("The geometric distribution models the number of trials needed to get the first success in a sequence of independent Bernoulli trials."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("p: probability of success on each trial"),
                    ui.tags.li("q: probability of failure on each trial"),
                    ui.tags.li("k: number of trials until first success")
                )
            )
        
        # Hypergeometric
        elif selected == "Hypergeometric":
            return ui.div(
                ui.h2("Hypergeometric Distribution"),
                ui.p("The Hypergeometric models...."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("N: total number of objects in the population"),
                    ui.tags.li("n: sample size"),
                    ui.tags.li("k: number of successes observed"),
                    ui.tags.li("K: number of defectives")
                )
            )
        
        # Negative Binomial 
        elif selected == "Negative Binomial":
            return ui.div(
                ui.h2("Negative Binomial"),
                ui.p("Negative Binomial models..."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("r: number of successes"),
                    ui.tags.li("p: probability of success"),
                    ui.tags.li("q: probability of failure"),
                    ui.tags.li("k: number of trials until rth success")
                    )
                )

        
        # --------------------------------- Continuous Distributions -----------------------------------
        
        # Uniform Section
        elif selected == "Uniform":
            return ui.div(
                ui.h2("Uniform Distribution"),
                ui.p("The Uniform distribution is a continuous probability distribution where all outcomes are equally likely."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("a: minimum value"),
                    ui.tags.li("b: maximum value"),
                    ui.tags.li("x: random variable"),
                    ui.tags.li("n: number of equally likely outcomes")
                )
            )
        
        # Exponential Section
        elif selected == "Exponential":
            return ui.div(
                ui.h2("Exponential Distribution"),
                ui.p("The Exponential distribution models the time between events in a Poisson point process."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("λ (lambda): rate of success"),
                    ui.tags.li("x: random variable")
                )
            )
        
        # Student Section
        elif selected == "Student":
            return ui.div(
                ui.h2("Student Distribution"),
                ui.p("The Student distribution models..."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("ν: degress of freedom"),
                    ui.tags.ul("x: random variable")
                )
            )
        
        # F Section
        elif selected == "F":
            return ui.div(
                ui.h2("F Distribution"),
                ui.p("The F distribution models..."),
                ui.p("Key parameters"),
                ui.tags.ul(
                    ui.tags.li("ν1: degress of freedom of the numerator"),
                    ui.tags.li("ν2: degress of freedom of the denominator"),
                    ui.tags.li("x: random variable")
                )
            )
        
        # Gamma Section
        elif selected == "Gamma":
            return ui.div(
                ui.h2("Gamma Distribution"),
                ui.p("The Gamma distribution models..."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("α: shape parameter"),
                    ui.tags.li("β: rate parameter"),
                    ui.tags.li("x: random variable")
                )
            )
        
        # Chi-Square Section
        elif selected == "Chi-square":
            return ui.div(
                ui.h2("Chi-square Distribution"),
                ui.p("The Chi-square distribution models..."),
                ui.p("Key parameters:"),
                ui.tags.ul(
                    ui.tags.li("ν: degrees of freedom"),
                    ui.tags.li("x: random variable")
                )
            )
        

        # ---------------------------------------------------------------------------------------
        
    # Distribution inputs
    @output
    @render.ui
    def distribution_inputs():
       """
       This allows the user to input the parameters for calculation
       """
       selected = input.specific_distribution()

       # ----------------------------- Discrete Inputs Section -----------------------------

       # Binomial Section
       if selected == "Binomial":
            return ui.div(
                ui.input_text("n", "Number of trials (n)"),
                ui.input_text("p", "Probability of success (p)"),
                ui.input_text("k", "Number of successes (k)"),
                ui.input_action_button("calculate", "Calculate")
            )
       
       # Poisson Section
       elif selected == "Poisson":
           return ui.div(
               ui.input_text("k", "Number of successes (k)"),
               ui.input_text("λ", "Average rate of success (λ)"),
               ui.input_action_button("calculate", "Calculate")
               )
       
       # Geometric Section
       elif selected == "Geometric":
           return ui.div(
               ui.input_text("p", "Probability of success (p)"),
               ui.input_text("q", "Probability of failure (q)"),
               ui.input_text("k", "Number of trials till first success (k)"),
               ui.input_action_button("calculate", "Calculate")
           )
       
       # Hypergeometric Section
       elif selected == "Hypergeometric":
           return ui.div(
               ui.input_text("N", "Total number of objects in the population (N)"),
               ui.input_text("K", "Number of defectives (K)"),
               ui.input_text("n", "Sample size (n)"),
               ui.input_text("k", "Number of successes observed (k)"),
               ui.input_action_button("calculate", "Calculate")
           )
       
       # Negative Binomial Section
       elif selected == "Negative Binomial":
           return ui.div(
               ui.input_text("r", "Number of successes (r)"),
               ui.input_text("p", "Probability of success (p)"),
               ui.input_text("q", "Probability of failure (q)"),
               ui.input_text("k", "Number of trials until rth success(k)"),
               ui.input_action_button("calculate", "Calculate")
           )
       
       
       # ----------------------------- Continuous Inputs Sections -----------------------------
       
       # Uniform Distribution Section
       elif selected == "Uniform":
           return ui.div(
               ui.input_text("a", "Minimum value (a)"),
               ui.input_text("b", "Maximum value (b)"),
               ui.input_text("x", "Random variable (x)"),
               ui.input_action_button("calculate", "Calculate")
           )
       
       # Exponential Section
       elif selected == "Exponential":
           return ui.div(
               ui.input_text("λ", "Rate of success (λ)"),
               ui.input_text("x", "Random variable(x)"),
               ui.input_action_button("calculate", "Calculate")
           )
       
       # Student Section
       elif selected == "Student":
           return ui.div(
               ui.input_text("ν", "Degrees of freedom (ν)"),
               ui.input_text("x", "Random Variable"),
               ui.input_action_button("calculate", "Calculate")
           )
       
       # F Section
       elif selected == "F":
           return ui.div(
               ui.input_text("ν1", "Degrees of freedom (ν1)"),
               ui.input_text("ν2", "Degrees of freedom (ν2)"),
               ui.input_text("x",  "Random variable (x)"),
               ui.input_action_button("calculate", "Calculate")
           )
       
       # Gamma Section
       elif selected == "Gamma":
           return ui.div(
               ui.input_text("α", "Shape parameter (α)"),
               ui.input_text("β", "Rate parameter (β)"),
               ui.input_text("x", "Random variable (x)"),
               ui.input_action_button("calculate", "Calculate")
           )
       
       # Chi-square Section
       elif selected == "Chi-square":
           return ui.div(
               ui.input_text("ν", "Degrees of freedom (ν)"),
               ui.input_text("x", "Random variable (x)"),
               ui.input_action_button("calculate", "Calculate")
           )

       # ---------------------------------------------------------------------------------------


    # Distribution results
    @output
    @render.ui
    @reactive.event(input.calculate)
    def distribution_results():
        """
        This function produces the results based on the inputs provided
        by the user over in distribution_inputs().
        """
        selected = input.specific_distribution()

        # ----------------------------- Discrete Sections --------------------------------

        # Binomial Section
        if selected == "Binomial":
            try:
                n = int(input.n())
                p = float(input.p())
                k = int(input.k())
                q = 1 - p
                
                dist_bin = BinomialDist(n, p, q, k)
                
                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {k}) = {dist_bin.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {k}) = {dist_bin.cdf():.6f}"),
                    ui.p(f"Mean: {dist_bin.mean:.6f}"),
                    ui.p(f"Variance: {dist_bin.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_bin.std_dev:.6f}")
                )
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
        
        # Poisson Section
        elif selected == "Poisson":
            try:
                λ = int(input.λ())
                k = int(input.k())

                dist_pois = PoissonDist(λ, k)

                return ui.div(
                     ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {k}) = {dist_pois.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {k}) = {dist_pois.cdf():.6f}"),
                    ui.p(f"Mean: {dist_pois.mean:.6f}"),
                    ui.p(f"Variance: {dist_pois.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_pois.std_dev:.6f}")
                )
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            
        # Geometric Section
        elif selected == "Geometric":
            try:
                p =  float(input.p())
                k = int(input.k())
                q = float(input.q())

                dist_geo = GeometricDist(p,q,k)

                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {k}) = {dist_geo.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {k}) = {dist_geo.cdf():.6f}"),
                    ui.p(f"Mean: {dist_geo.mean:.6f}"),
                    ui.p(f"Variance: {dist_geo.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_geo.std_dev:.6f}")
                )
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            
        # Hypergeometric Section
        elif selected == "Hypergeometric":
            try:
                N = int(input.N())
                K = int(input.K())
                n = int(input.n())
                k = int(input.k())

                dist_hypergeo = HypergeometricDist(N,n,K,k)

                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {k}) = {dist_hypergeo.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {k}) = {dist_hypergeo.cdf():.6f}"),
                    ui.p(f"Mean: {dist_hypergeo.mean:.6f}"),
                    ui.p(f"Variance: {dist_hypergeo.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_hypergeo.std_dev:.6f}")
                )
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            
        # Negative Binomial Section
        elif selected == "Negative Binomial":
            try:
                r = int(input.r())
                p = float(input.p())
                q = float(input.q())
                k = int(input.k())

                dist_negative = NegativeBinomialDist(r,p,q,k)

                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {k}) = {dist_negative.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {k}) = {dist_negative.cdf():.6f}"),
                    ui.p(f"Mean: {dist_negative.mean:.6f}"),
                    ui.p(f"Variance: {dist_negative.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_negative.std_dev:.6f}")
                )
            
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            

        # ----------------------------- Continuous Sections -------------------------------------

        # Uniform Section
        elif selected == "Uniform":
            try:
                a = float(input.a())
                b = float(input.b())
                x = float(input.x())

                dist_uniform = UniformDist(a,b,x)
            
                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {x}) = {dist_uniform.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {x}) = {dist_uniform.cdf():.6f}"),
                    ui.p(f"Mean: {dist_uniform.mean:.6f}"),
                    ui.p(f"Variance: {dist_uniform.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_uniform.std_dev:.6f}")
                )
            
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            
        # Exponential Section
        elif selected == "Exponential":
            try:
                λ = float(input.λ())
                x = float(input.x())

                dist_exp = ExponentialDist(λ,x)

                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {x}) = {dist_exp.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {x}) = {dist_exp.cdf():.6f}"),
                    ui.p(f"Mean: {dist_exp.mean:.6f}"),
                    ui.p(f"Variance: {dist_exp.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_exp.std_dev:.6f}")
                )
            
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            
        # Student Section
        elif selected == "Student":
            try:
                ν = float(input.ν())
                x = float(input.x())

                dist_student = StudentDist(ν,x)

                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {x}) = {dist_student.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {x}) = {dist_student.cdf():.6f}"),
                    ui.p(f"Mean: {dist_student.mean:.6f}"),
                    ui.p(f"Variance: {dist_student.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_student.std_dev:.6f}")
                )
            
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            
        # F Section
        elif selected == "F":
            try:
                ν1 = float(input.ν1())
                ν2 = float(input.ν2())
                x = float(input.x())

                dist_f = FDist(ν1, ν2, x)

                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {x}) = {dist_f.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {x}) = {dist_f.cdf():.6f}"),
                    ui.p(f"Mean: {dist_f.mean:.6f}"),
                    ui.p(f"Variance: {dist_f.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_f.std_dev:.6f}")
                )
            
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            
        # Gamma Section
        elif selected == "Gamma":
            try:
                α = float(input.α())
                β = float(input.β())
                x = float(input.x())

                dist_gamma = GammaDist(α, β, x)

                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {x}) = {dist_gamma.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {x}) = {dist_gamma.cdf():.6f}"),
                    ui.p(f"Mean: {dist_gamma.mean:.6f}"),
                    ui.p(f"Variance: {dist_gamma.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_gamma.std_dev:.6f}")
                )
            
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
            
        # Chi_square Section
        elif selected == "Chi-square":
            try:
                ν = float(input.ν())
                x = float(input.x())

                dist_chi = ChiSquaredDist(ν,x)

                return ui.div(
                    ui.h3("Results:"),
                    ui.p(f"PMF: P(X = {x}) = {dist_chi.pmf():.6f}"),
                    ui.p(f"CDF: P(X ≤ {x}) = {dist_chi.cdf():.6f}"),
                    ui.p(f"Mean: {dist_chi.mean:.6f}"),
                    ui.p(f"Variance: {dist_chi.variance:.6f}"),
                    ui.p(f"Standard Deviation: {dist_chi.std_dev:.6f}")
                )
            
            except ValueError as e:
                return ui.p(f"Error: {str(e)}", style= "color: red;")
            except Exception as e:
                return ui.p(f"An unexpected error occurred: {str(e)}", style="color: red;")
        # ---------------------------------------------------------------------------------------


# Runs Application
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()