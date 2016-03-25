package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.RealMatrix;

import defs.Dataset;
import defs.Hypers;
import defs.Params;
import defs.SoRecParams;

class SoRec_Cal extends RecSysCal {
	
	Dataset ds; 
	Hypers hypers;
	
	public SoRec_Cal(Dataset ds, Hypers hypers) {
		super(ds);
		this.ds = ds;
		this.hypers = hypers;
	}

	@Override
	double objValue(Params params) {
		SoRecParams soRecParams = (SoRecParams) params;
		
		RealMatrix ratingErrs = calRatingErrors(soRecParams);
		RealMatrix edgeWeightErrs = calEdgeWeightErrors(soRecParams);
		double value = sqFrobNorm(ratingErrs) + hypers.weightLambda * sqFrobNorm(edgeWeightErrs);
		double regPart = sqFrobNorm(soRecParams.topicUser) + sqFrobNorm(soRecParams.topicItem) + sqFrobNorm(soRecParams.zMatrix);
		value += hypers.topicLambda * regPart;  
		
		return value;
	}

	@Override
	RealMatrix estRatings(Params params) {
		RealMatrix estRatings = params.topicUser.transpose().multiply(params.topicItem);
		return estRatings;
	}

	@Override
	RealMatrix calRatingErrors(Params params) {
		RealMatrix estimated_ratings = estRatings(params);
		RealMatrix bounded_ratings = UtilFuncs.cutoff(estimated_ratings);
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);
		return rating_errors;
	}
	
	RealMatrix estWeights(SoRecParams params) {
		RealMatrix estimated_weights = params.topicUser.transpose().multiply(params.zMatrix);
		return estimated_weights;
	}
	
	RealMatrix calEdgeWeightErrors(SoRecParams params) {
		RealMatrix estimated_weights = estWeights(params);
		RealMatrix bounded_weights = UtilFuncs.cutoff(estimated_weights);
		RealMatrix edge_weight_errors = ErrorCal.edgeWeightErrors(bounded_weights, ds.edge_weights);
		return edge_weight_errors;
	}
	
	private double sqFrobNorm(RealMatrix matrix) {
		return UtilFuncs.square(matrix.getFrobeniusNorm());
	}

	public RealMatrix calRatingErrors(RealMatrix estimated_ratings, RealMatrix ratings) {
		
		RealMatrix bounded_ratings = UtilFuncs.cutoff(estimated_ratings);
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);
		return rating_errors;
	}

}
