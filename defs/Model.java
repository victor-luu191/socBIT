package defs;

import java.util.Optional;

import core.RecSysCal;

public class Model {
	
	public Params learnedParams;
	public RecSysCal calculator; // or estimator
	
	// residuals
	public double rating_rmse;
	public double trust_rmse;
	
	public double objValue;
	
	public Model(Params learnedParams, RecSysCal calculator, double rating_rmse, Optional<Double> optTrust_rmse, double objValue) {
		this.learnedParams = learnedParams;
		this.calculator = calculator;
		
		this.rating_rmse = rating_rmse;
		this.trust_rmse = optTrust_rmse.orElse(Double.NaN);
		this.objValue = objValue;
	}

	public String toErrString() {
		return rating_rmse + "," + trust_rmse + "," + objValue;
	}
	
}
