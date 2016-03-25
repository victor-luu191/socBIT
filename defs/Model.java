package defs;

import java.util.Optional;

import core.RecSysCal;

public class Model {
	
	public Params learnedParams;
	public RecSysCal calculator; // or estimator
	// residuals
	public double ratingErr;
	public double edgeWeightErr;
	
	public double objValue;
	
	public Model(Params learnedParams, RecSysCal calculator, double ratingErr, Optional<Double> optEdgeWeightErr, double objValue) {
		this.learnedParams = learnedParams;
		this.calculator = calculator;
		
		this.ratingErr = ratingErr;
		this.edgeWeightErr = optEdgeWeightErr.orElse(Double.NaN);
		this.objValue = objValue;
	}

	public String toErrString() {
		return ratingErr + "," + edgeWeightErr + "," + objValue;
	}
	
}
