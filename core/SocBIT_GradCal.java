package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Params;
import defs.SocBIT_Params;

public class SocBIT_GradCal extends GradCal {

	private SocBIT_Cal calculator;
	private RealMatrix estimated_weights;
	private RealMatrix edge_weight_errors;
	
	public SocBIT_GradCal(Trainer trainer) {
		numTopic = trainer.numTopic;
		ds = trainer.ds;
		hypers = trainer.hypers;
		calculator = new SocBIT_Cal(ds, hypers);
	}
	
	@Override
	Params calculate(Params params) {
		
		SocBIT_Params castParams = (SocBIT_Params) params;
		
		estimated_ratings = calculator.estRatings(castParams);
		rating_errors = calculator.calRatingErrors(castParams);
		
		estimated_weights = calculator.estWeights(castParams);
		RealMatrix bounded_weights = UtilFuncs.cutoff(estimated_weights);
		edge_weight_errors = ErrorCal.edgeWeightErrors(bounded_weights, ds.edge_weights);	// estimated_weights
		
		SocBIT_Params grad = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, this.numTopic);
		// gradients for users
		for (int u = 0; u < ds.numUser; u++) {
			grad.userDecisionPrefs[u] = userDecisionPrefDiff(castParams, u);
			grad.topicUser.setColumnVector(u, calUserTopicGrad(params, u));
			grad.brandUser.setColumnVector(u, userBrandGrad(castParams, u));
			// do smth here to debug
		}
		
		// gradients for items
		for (int i = 0; i < ds.numItem; i++) {
			grad.topicItem.setColumnVector(i, calItemTopicGrad(params, i));
			grad.brandItem.setColumnVector(i, itemBrandGrad(castParams, i));
		}
		
		return grad;
	}
	
	@Override
	RealVector calItemTopicGrad(Params params, int itemIndex) {
		
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
	
	@Override
	RealVector calUserTopicGrad(Params params, int u) {
		
		RealVector userTopicFeats = params.topicUser.getColumnVector(u);
		double topicLambda = hypers.topicLambda;
		RealVector topicGrad = userTopicFeats.mapMultiply(topicLambda);
		
		// component wrt rating errors
		RealVector rating_sum = new ArrayRealVector(numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector curItemTopicFeat = params.topicItem.getColumnVector(i);
			double ratingLogisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
			double rError = rating_errors.getEntry(u, i);
			RealVector modified_topicFeat = curItemTopicFeat.mapMultiply(rError*ratingLogisDiff);
			rating_sum = rating_sum.add(modified_topicFeat);
		}
		
		// component wrt error of edge weight estimation 
		RealVector edge_weight_sum = new ArrayRealVector(numTopic);
		for (int v = 0; v < ds.numUser; v++) {
			RealVector curUserTopicFeat = params.topicUser.getColumnVector(v);
			double weightLogisDiff = UtilFuncs.logisDiff(estimated_weights.getEntry(u, v));
			double trustErr = edge_weight_errors.getEntry(u, v);
			RealVector modified_topicFeat = curUserTopicFeat.mapMultiply(trustErr*weightLogisDiff);
			edge_weight_sum = edge_weight_sum.add(modified_topicFeat);

		}

		double weightLambda = hypers.weightLambda;
		RealVector bigSum = rating_sum.add(edge_weight_sum.mapMultiply(weightLambda));
		SocBIT_Params castParams = (SocBIT_Params) params;
		double uDecPref = castParams.userDecisionPrefs[u];
		topicGrad = topicGrad.add(bigSum.mapMultiply(uDecPref)); 	// see Eqn. 26
		return topicGrad;
	}

	RealVector itemBrandGrad(SocBIT_Params params, int itemIndex) {

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
	
	RealVector userBrandGrad(SocBIT_Params params, int u) {
		
		RealVector curUserBrandFeat = params.brandUser.getColumnVector(u);
		double brandLambda = hypers.brandLambda;
		RealVector nextBrandGrad = curUserBrandFeat.mapMultiply(brandLambda);
		// component wrt rating errors
		RealVector rating_sum = calRatingSum(params, u);
		
		// component wrt error of edge weight estimation
		RealVector edge_weight_sum = calEdgeWeightSum(params, u);
		
		double weightLambda = hypers.weightLambda;
		RealVector bigSum = rating_sum.add(edge_weight_sum.mapMultiply(weightLambda));
		
		nextBrandGrad = nextBrandGrad.add(bigSum.mapMultiply(1 - params.userDecisionPrefs[u]));	// see Eqn. 27
		return nextBrandGrad;
	}

	private RealVector calEdgeWeightSum(SocBIT_Params params, int u) {
		RealVector edge_weight_sum = new ArrayRealVector(ds.numBrand);
		for (int v = 0; v < ds.numUser; v++) {
			RealVector vBrandFeat = params.brandUser.getColumnVector(v);
			double weightLogisDiff = UtilFuncs.logisDiff(estimated_weights.getEntry(u, v));
			RealVector modified_BrandFeat = vBrandFeat.mapMultiply(edge_weight_errors.getEntry(u, v)).mapMultiply(weightLogisDiff);
			edge_weight_sum = edge_weight_sum.add(modified_BrandFeat);
		}
		return edge_weight_sum;
	}

	private RealVector calRatingSum(SocBIT_Params params, int u) {
		RealVector rating_sum = new ArrayRealVector(ds.numBrand);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector curItemBrandFeat = params.brandItem.getColumnVector(i);
			double ratingLogisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
			RealVector modified_brandFeat = curItemBrandFeat.mapMultiply(rating_errors.getEntry(u, i)).mapMultiply(ratingLogisDiff);
			rating_sum = rating_sum.add(modified_brandFeat);
		}
		return rating_sum;
	}
	
	double userDecisionPrefDiff(SocBIT_Params params, int u) {
		
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
			double entry = estimated_ratings.getEntry(u, i);
			double ratingLogisDiff = UtilFuncs.logisDiff(entry);
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
		decisionPrefDiff += bigSum;
		return decisionPrefDiff;
	}
}
