package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.*;

import defs.Dataset;
import defs.Hypers;

class SocBIT_Calculator {
	
	private SocBIT_Params params;
	
	/**
	 * fields derived from {@code params}
	 */
	private int numUser;
	private DiagonalMatrix decisionPrefs;
	private RealMatrix idMat;

	public SocBIT_Calculator(SocBIT_Params params) {
		this.params = params;
		decisionPrefs = new DiagonalMatrix(params.userDecisionPrefs);
		numUser = params.brandUser.getColumnDimension();
		idMat = MatrixUtils.createRealIdentityMatrix(numUser);
	}
	
	RealMatrix estRatings() {
		
		RealMatrix topicRatings = decisionPrefs.multiply(params.topicUser.transpose()).multiply(params.topicItem);
		RealMatrix brandRatings = idMat.subtract(decisionPrefs).multiply(params.brandUser.transpose()).multiply(params.brandItem);
		RealMatrix est_ratings =  topicRatings.add(brandRatings); 
		return est_ratings;
	}
	
	RealMatrix estWeights() {
		
		RealMatrix topicWeights = decisionPrefs.multiply(params.topicUser.transpose()).multiply(params.topicUser);
		RealMatrix brandWeights = idMat.subtract(decisionPrefs).multiply(params.brandUser.transpose()).multiply(params.brandUser);
		RealMatrix est_edge_weights = topicWeights.add(brandWeights);
		return est_edge_weights;
	}

	RealMatrix comp_ratingErrors(RealMatrix obsRatings) {
		RealMatrix estimated_ratings = estRatings();
		RealMatrix bounded_ratings = UtilFuncs.bound(estimated_ratings);
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, obsRatings);
		return rating_errors;
	}
	
	RealMatrix comp_edgeWeightErrors(RealMatrix obsEdgeWeights) {
		RealMatrix estimated_weights = estWeights();
		RealMatrix bounded_weights = UtilFuncs.bound(estimated_weights);
		RealMatrix edge_weight_errors = ErrorCal.edgeWeightErrors(bounded_weights, obsEdgeWeights);
		return edge_weight_errors;
	}

	
}
