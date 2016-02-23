package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;

class STE_Cal extends RecSysCal {
	
	Dataset ds; 
	Hypers hypers;
	
	public STE_Cal(Dataset ds, Hypers hypers) {
		super();
		this.ds = ds;
		this.hypers = hypers;
	}

	/**
	 * Estimated rating r_{u,i} by STE (social trust ensemble) model in Hao Ma paper: Learning to Recommend with Social Trust Ensemble
	 * @param u
	 * @param i
	 * @param edge_weights
	 * @return
	 */
	double estOneRating(int u, int i, Params params) {
		
		RealVector userTopicFeat = params.topicUser.getColumnVector(u);
		RealVector itemTopicFeat = params.topicItem.getColumnVector(i);
		double personal_rating = userTopicFeat.dotProduct(itemTopicFeat);
		
		double neighbor_rating = 0;
		for (int v = 0; v < ds.numUser; v++)  {
			RealVector v_topicFeat = params.topicUser.getColumnVector(v);
			neighbor_rating += ds.edge_weights.getEntry(v, u) * v_topicFeat.dotProduct(itemTopicFeat);
		}
		return hypers.alpha*personal_rating + (1 - hypers.alpha)*neighbor_rating;
	}

	RealMatrix estRatings(Params params) {
		RealMatrix estimated_ratings = new Array2DRowRealMatrix(ds.numUser, ds.numItem);
		for (int u = 0; u < ds.numUser; u++) {
			for (int i = 0; i < ds.numItem; i++) {
				estimated_ratings.setEntry(u, i, estOneRating(u, i, params));
			}
		}
		return estimated_ratings;
	}

	@Override
	double objValue(Params params) {
		
//		System.out.println("calculating value of objective function ...");
		double userFeatsNorm = params.topicUser.getFrobeniusNorm();
		double itemFeatsNorm = params.topicItem.getFrobeniusNorm();
		double val = hypers.topicLambda * (UtilFuncs.square(userFeatsNorm) + UtilFuncs.square(itemFeatsNorm));	// regularized part
		
		RealMatrix rating_errors = calRatingErrors(params);
		val += UtilFuncs.square(rating_errors.getFrobeniusNorm());
		return val;
	}

	private RealMatrix calRatingErrors(Params params) {
		
//		STE_Calculator calculator = new STE_Calculator(params, alpha, ds.edge_weights);
		RealMatrix estimated_ratings = estRatings(params);
		RealMatrix bounded_ratings = UtilFuncs.bound(estimated_ratings);
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);
		return rating_errors;
	}
}
