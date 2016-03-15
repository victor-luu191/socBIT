package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;
import defs.Params;

class STE_Cal extends RecSysCal {
	
	Dataset ds; 
	Hypers hypers;
	
	public STE_Cal(Dataset ds, Hypers hypers) {
		super(ds);
		this.ds = ds;
		this.hypers = hypers;
	}

	@Override
	double objValue(Params params) {
		
//		System.out.println("calculating value of objective function ...");
		
		RealMatrix rating_errors = calRatingErrors(params);
		double sqErr = UtilFuncs.square(rating_errors.getFrobeniusNorm());
		return sqErr + regularization(params) ;
	}

	double regularization(Params params) {
		double userFeatsNorm = params.topicUser.getFrobeniusNorm();
		double itemFeatsNorm = params.topicItem.getFrobeniusNorm();
		double val = hypers.topicLambda * (UtilFuncs.square(userFeatsNorm) + UtilFuncs.square(itemFeatsNorm));
		return val;
	}

	RealMatrix estRatings(Params params) {
		
		for (int u = 0; u < ds.numUser; u++) {
			for (int i = 0; i < ds.numItem; i++) {
				estimated_ratings.setEntry(u, i, estOneRating(u, i, params));
			}
		}
		return estimated_ratings;
	}

	RealMatrix calRatingErrors(Params params) {
		
//		STE_Calculator calculator = new STE_Calculator(params, alpha, ds.edge_weights);
		
		RealMatrix bounded_ratings = UtilFuncs.cutoff(estimated_ratings);
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);
		return rating_errors;
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
}
