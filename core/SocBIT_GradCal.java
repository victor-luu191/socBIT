package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class SocBIT_GradCal extends GradCal {

	public SocBIT_GradCal(Trainer trainer) {
		super(trainer);
	}

	@Override
	Params calculate(Params params) {
		
		SocBIT_Cal socBIT_Estimator = new SocBIT_Cal(ds, hypers);
		SocBIT_Params castParams = (SocBIT_Params) params;
		estimated_ratings = socBIT_Estimator.estRatings(castParams);
		RealMatrix bounded_ratings = UtilFuncs.bound(estimated_ratings);
		
		estimated_weights = socBIT_Estimator.estWeights(castParams);
		RealMatrix bounded_weights = UtilFuncs.bound(estimated_weights);
		
		RealMatrix edge_weight_errors = ErrorCal.edgeWeightErrors(bounded_weights, ds.edge_weights);	// estimated_weights
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);					// estimated_ratings
		
		SocBIT_Params grad = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, this.numTopic);
		// gradients for users
		for (int u = 0; u < ds.numUser; u++) {
			grad.userDecisionPrefs[u] = userDecisionPrefDiff(castParams, u, rating_errors, edge_weight_errors);
			grad.topicUser.setColumnVector(u, userTopicGrad(params, u, rating_errors, edge_weight_errors));
			grad.brandUser.setColumnVector(u, userBrandGrad(castParams, u, rating_errors, edge_weight_errors));
		}
		
		// gradients for items
		for (int i = 0; i < ds.numItem; i++) {
			grad.topicItem.setColumnVector(i, itemTopicGrad(params, i, rating_errors));
			grad.brandItem.setColumnVector(i, itemBrandGrad(castParams, i, rating_errors));
		}
		
		return grad;
	}
	
	@Override
	RealVector itemTopicGrad(Params params, int itemIndex, RealMatrix rating_errors) {
		
		SocBIT_Params castParams = (SocBIT_Params) params;
		RealVector itemTopicFeats = castParams.topicItem.getColumnVector(itemIndex);
		double topicLambda = hypers.topicLambda;
		RealVector topicGrad = itemTopicFeats.mapMultiply(topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int u = 0; u < ds.numUser; u++) {
			double w = castParams.userDecisionPrefs[u];
			double weighted_rating_err = w * rating_errors.getEntry(u, itemIndex);
			RealVector userTopicFeat = castParams.topicUser.getColumnVector(u);
			double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, itemIndex));
			sum = sum.add(userTopicFeat.mapMultiply(weighted_rating_err).mapMultiply(logisDiff));
		}
		
		topicGrad = topicGrad.add(sum);
		return topicGrad;
	}

	private RealVector userTopicGrad(SocBIT_Params params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {

		RealVector userTopicFeats = params.topicUser.getColumnVector(u);
		double topicLambda = hypers.topicLambda;
		RealVector topicGrad = userTopicFeats.mapMultiply(topicLambda);
		// component wrt rating errors
		RealVector rating_sum = new ArrayRealVector(numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector curItemTopicFeat = params.topicItem.getColumnVector(i);
			double ratingLogisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
			RealVector modified_topicFeat = curItemTopicFeat.mapMultiply(rating_errors.getEntry(u, i)).mapMultiply(ratingLogisDiff);
			rating_sum = rating_sum.add(modified_topicFeat);
		}
		// component wrt error of edge weight estimation 
		RealVector edge_weight_sum = new ArrayRealVector(numTopic);
		for (int v = 0; v < ds.numUser; v++) {
			RealVector curUserTopicFeat = params.topicUser.getColumnVector(v);
			double weightLogisDiff = UtilFuncs.logisDiff(estimated_weights.getEntry(u, v));
			RealVector modified_topicFeat = curUserTopicFeat.mapMultiply(edge_weight_errors.getEntry(u, v)).mapMultiply(weightLogisDiff);
			edge_weight_sum = edge_weight_sum.add(modified_topicFeat);

		}

		double weightLambda = hypers.weightLambda;
		RealVector bigSum = rating_sum.add(edge_weight_sum.mapMultiply(weightLambda));
		topicGrad = topicGrad.add(bigSum.mapMultiply(params.userDecisionPrefs[u])); 	// see Eqn. 26
		return topicGrad;
	}

	RealVector itemBrandGrad(SocBIT_Params params, int itemIndex, RealMatrix rating_errors) {

		RealVector curBrandGrad = params.brandItem.getColumnVector(itemIndex);
		double brandLambda = hypers.brandLambda;
		RealVector nextBrandGrad = curBrandGrad.mapMultiply(brandLambda);

		RealVector sum = new ArrayRealVector(ds.numBrand);
		for (int u = 0; u < ds.numUser; u++) {
			double w = 1 - params.userDecisionPrefs[u];
			double weighted_rating_err = w * rating_errors.getEntry(u, itemIndex);
			RealVector userBrandFeat = params.brandUser.getColumnVector(u);
			double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, itemIndex));
			sum = sum.add(userBrandFeat.mapMultiply(weighted_rating_err).mapMultiply(logisDiff));
		}
		nextBrandGrad = nextBrandGrad.add(sum);
		return nextBrandGrad;
	}
	
	RealVector userBrandGrad(SocBIT_Params params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {
		
		RealVector curBrandGrad = params.brandUser.getColumnVector(u);
		double brandLambda = hypers.brandLambda;
		RealVector nextBrandGrad = curBrandGrad.mapMultiply(brandLambda);
		// component wrt rating errors
		RealVector rating_sum = new ArrayRealVector(ds.numBrand);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector curItemBrandFeat = params.brandItem.getColumnVector(i);
			double ratingLogisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
			RealVector modified_brandFeat = curItemBrandFeat.mapMultiply(rating_errors.getEntry(u, i)).mapMultiply(ratingLogisDiff);
			rating_sum = rating_sum.add(modified_brandFeat);
		}
		
		// component wrt error of edge weight estimation
		RealVector edge_weight_sum = new ArrayRealVector(ds.numBrand);
		for (int v = 0; v < ds.numUser; v++) {
			RealVector curUserBrandFeat = params.brandUser.getColumnVector(v);
			double weightLogisDiff = UtilFuncs.logisDiff(estimated_weights.getEntry(u, v));
			RealVector modified_BrandFeat = curUserBrandFeat.mapMultiply(edge_weight_errors.getEntry(u, v)).mapMultiply(weightLogisDiff);
			edge_weight_sum = edge_weight_sum.add(modified_BrandFeat);
		}
		
		double weightLambda = hypers.weightLambda;
		RealVector bigSum = rating_sum.add(edge_weight_sum.mapMultiply(weightLambda));
		
		nextBrandGrad = nextBrandGrad.add(bigSum.mapMultiply(1 - params.userDecisionPrefs[u]));	// see Eqn. 27
		return nextBrandGrad;
	}
	
	double userDecisionPrefDiff(SocBIT_Params params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {
		
		double userDecisionPref = params.userDecisionPrefs[u];
		double decisionLambda = hypers.decisionLambda;
		double decisionPrefDiff = decisionLambda * (userDecisionPref - 0.5);
		
		RealVector theta_u = params.topicUser.getColumnVector(u);
		RealVector beta_u = params.brandUser.getColumnVector(u);
		
		double rating_sum = 0;
		for (int i = 0; i < ds.numItem; i++) {
			RealVector theta_i = params.topicItem.getColumnVector(i);
			RealVector beta_i = params.brandItem.getColumnVector(i);
			double topicSim = theta_u.dotProduct(theta_i);
			double brandSim = beta_u.dotProduct(beta_i);
			double ratingLogisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
			rating_sum += rating_errors.getEntry(u, i) * (topicSim - brandSim) * ratingLogisDiff;
		}
		
		double edge_weight_sum = 0;
		for (int v = 0; v < ds.numUser; v++) {
			RealVector theta_v = params.topicUser.getColumnVector(v);
			RealVector beta_v = params.brandUser.getColumnVector(v);
			double topicSim = theta_u.dotProduct(theta_v);
			double brandSim = beta_u.dotProduct(beta_v);
			double weightLogisDiff = UtilFuncs.logisDiff(estimated_weights.getEntry(u, v));
			edge_weight_sum += edge_weight_errors.getEntry(u, v) * (topicSim - brandSim) * weightLogisDiff;
		}
		
		double weightLambda = hypers.weightLambda;
		double bigSum = rating_sum + weightLambda * edge_weight_sum;
		decisionPrefDiff = decisionPrefDiff + bigSum;
		return decisionPrefDiff;
	}
}
