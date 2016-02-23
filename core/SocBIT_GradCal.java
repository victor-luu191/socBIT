package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.RealMatrix;

public class SocBIT_GradCal extends GradCal {

	public SocBIT_GradCal(Trainer trainer) {
		super(trainer);
	}

	@Override
	Params calculate(Params params) {
		
		SocBIT_Cal socBIT_Estimator = new SocBIT_Cal(ds, hypers);
		estimated_ratings = socBIT_Estimator.estRatings((SocBIT_Params) params);
		RealMatrix bounded_ratings = UtilFuncs.bound(estimated_ratings);
		
		estimated_weights = socBIT_Estimator.estWeights((SocBIT_Params) params);
		RealMatrix bounded_weights = UtilFuncs.bound(estimated_weights);
		
		RealMatrix edge_weight_errors = ErrorCal.edgeWeightErrors(bounded_weights, ds.edge_weights);	// estimated_weights
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);					// estimated_ratings
		
		SocBIT_Params grad = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, this.numTopic);
		// gradients for users
		for (int u = 0; u < ds.numUser; u++) {
			grad.userDecisionPrefs[u] = userDecisionPrefDiff(params, u, rating_errors, edge_weight_errors);
			grad.topicUser.setColumnVector(u, userTopicGrad(params, u, rating_errors, edge_weight_errors));
			grad.brandUser.setColumnVector(u, userBrandGrad(params, u, rating_errors, edge_weight_errors));
		}
		
		// gradients for items
		for (int i = 0; i < ds.numItem; i++) {
			grad.topicItem.setColumnVector(i, itemTopicGrad(params, i, rating_errors));
			grad.brandItem.setColumnVector(i, itemBrandGrad(params, i, rating_errors));
		}
		
		return grad;
	}

}
