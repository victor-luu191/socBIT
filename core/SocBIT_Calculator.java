package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.*;

import defs.Dataset;
import defs.Hypers;

class SocBIT_Calculator {
	
	Dataset ds; 
	Hypers hypers;
	
	private SocBIT_Params params;
	
	/**
	 * fields derived from {@code params}
	 */
	private DiagonalMatrix decisionPrefs;
	private RealMatrix idMat;

	public SocBIT_Calculator(Dataset ds, Hypers hypers) {
		this.ds = ds;
		this.hypers = hypers;
		
		decisionPrefs = new DiagonalMatrix(params.userDecisionPrefs);
		idMat = MatrixUtils.createRealIdentityMatrix(ds.numUser);
	}
	
	RealMatrix estRatings(SocBIT_Params params) {
		
		RealMatrix topicRatings = decisionPrefs.multiply(params.topicUser.transpose()).multiply(params.topicItem);
		RealMatrix brandRatings = idMat.subtract(decisionPrefs).multiply(params.brandUser.transpose()).multiply(params.brandItem);
		RealMatrix est_ratings =  topicRatings.add(brandRatings); 
		return est_ratings;
	}
	
	RealMatrix estWeights(SocBIT_Params params) {
		
		RealMatrix topicWeights = decisionPrefs.multiply(params.topicUser.transpose()).multiply(params.topicUser);
		RealMatrix brandWeights = idMat.subtract(decisionPrefs).multiply(params.brandUser.transpose()).multiply(params.brandUser);
		RealMatrix est_edge_weights = topicWeights.add(brandWeights);
		return est_edge_weights;
	}

	RealMatrix calRatingErrors(SocBIT_Params params) {
		RealMatrix estimated_ratings = estRatings(params);
		RealMatrix bounded_ratings = UtilFuncs.bound(estimated_ratings);
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);
		return rating_errors;
	}
	
	RealMatrix calEdgeWeightErrors(SocBIT_Params params) {
		RealMatrix estimated_weights = estWeights(params);
		RealMatrix bounded_weights = UtilFuncs.bound(estimated_weights);
		RealMatrix edge_weight_errors = ErrorCal.edgeWeightErrors(bounded_weights, ds.edge_weights);
		return edge_weight_errors;
	}

	double objValue(SocBIT_Params params) {

		RealMatrix rating_errors = calRatingErrors(params);
		RealMatrix edge_weight_errors = calEdgeWeightErrors(params);

		double val = sqFrobNorm(rating_errors);
		val += hypers.weightLambda * sqFrobNorm(edge_weight_errors);
		val += hypers.topicLambda * ( sqFrobNorm(params.topicUser) + sqFrobNorm(params.topicItem) );
		val += hypers.brandLambda * ( sqFrobNorm(params.brandUser) + sqFrobNorm(params.brandItem) );
		for (int u = 0; u < ds.numUser; u++) {
			val += hypers.decisionLambda * UtilFuncs.square(params.userDecisionPrefs[u] - 0.5);
		}
		return val;
	}
	
	// squared Frobenius Norm
	private double sqFrobNorm(RealMatrix matrix) {
		return UtilFuncs.square(matrix.getFrobeniusNorm());
	}

}
