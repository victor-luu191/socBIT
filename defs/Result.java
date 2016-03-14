package defs;

import java.util.Optional;

public class Result {
	
	public Params learnedParams;
	public double ratingErr;
	public double edgeWeightErr;
	public double objValue;
	
	public Result(Params learnedParams, double ratingErr, Optional<Double> optEdgeWeightErr, double objValue) {
		this.learnedParams = learnedParams;
		this.ratingErr = ratingErr;
		this.edgeWeightErr = optEdgeWeightErr.orElse(Double.NaN);
		this.objValue = objValue;
	}

	public String toErrString() {
		return ratingErr + "," + edgeWeightErr + "," + objValue;
	}
	
}
