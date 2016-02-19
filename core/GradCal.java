package core;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;

public class GradCal {
	
	private Hypers hypers;
	// derived fields
	private int numTopic;
	Dataset ds;
	private RealMatrix estimated_ratings;
	private RealMatrix estimated_weights;
	private double alpha;	// tuning parameter of STE only, control how much each user trusts himself vs. trust friends
	
	public GradCal(GD_Trainer trainer) {
		
		numTopic = trainer.numTopic;
		ds = trainer.ds;
		hypers = trainer.hypers;
	}
	
	/**
	 * Main work is here !!!
	 * Compute the gradient at a given set of params by SocBIT model; by computing all the components of the gradient
	 * @param params
	 * @return the complete gradient with all its components
	 */
	SocBIT_Params socBIT_Grad(SocBIT_Params params) {
		
		SocBIT_Estimator socBIT_Estimator = new SocBIT_Estimator(params);
		estimated_ratings = socBIT_Estimator.estRatings();
		RealMatrix bounded_ratings = UtilFuncs.bound(estimated_ratings);
		
		estimated_weights = socBIT_Estimator.estWeights();
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
	
	Params ste_Grad(Params params) {
		
		STE_estimator ste_estimator = new STE_estimator(params, alpha, ds.edge_weights);
		estimated_ratings = ste_estimator.estRatings();
		RealMatrix bounded_ratings = UtilFuncs.bound(estimated_ratings);
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);
		RealMatrix edge_weight_errors = new Array2DRowRealMatrix();	// just a dummy matrix as STE model don't estimate edge weights
		
		Params grad = new Params(ds.numUser, ds.numItem, numTopic);
		// gradients for users
		for (int u = 0; u < ds.numUser; u++) {
			RealVector userTopicGrad = userTopicGrad(params, u, rating_errors, edge_weight_errors);
			grad.topicUser.setColumnVector(u, userTopicGrad);
		}
		
		// gradients for items
		for (int i = 0; i < ds.numItem; i++) {
			grad.topicItem.setColumnVector(i, itemTopicGrad(params, i, rating_errors));
		}
		return grad;
	}
	
	/**
	 * @param itemIndex
	 * @param cParams
	 * @param topicLambda
	 * @param ratingErr: errors of estimating ratings of the item
	 * @return
	 */
	RealVector itemTopicGrad(Params params, int itemIndex, RealMatrix rating_errors) {
		
		RealVector topicGrad = new ArrayRealVector(numTopic);
		if (params instanceof SocBIT_Params) {
			SocBIT_Params tmp_params = new SocBIT_Params((SocBIT_Params) params);
			topicGrad = socBIT_itemTopicGrad(tmp_params, itemIndex, rating_errors);
		} 
		else {// simpler parameters with only topic features
			topicGrad = ste_itemTopicGrad(params, itemIndex, rating_errors);
		}
		
		return topicGrad;
	}

	/**
	 * @param u
	 * @param cParams
	 * @param topicLambda
	 * @param ratingErr
	 * @param weightLambda 
	 * @param strengthErr: errors in estimating strength of relationships of the user 
	 * @return
	 */
	RealVector userTopicGrad(Params params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {
		
		RealVector topicGrad = new ArrayRealVector(numTopic);
		if (params instanceof SocBIT_Params) {
			SocBIT_Params tmp_params = new SocBIT_Params((SocBIT_Params) params);
			topicGrad = socBIT_userTopicGrad(tmp_params, u, rating_errors, edge_weight_errors);
		} 
		else {// simpler parameters with only topic features
			topicGrad = ste_userTopicGrad(params, u, rating_errors);
		}
		
		return topicGrad;
	}
	
	private RealVector socBIT_itemTopicGrad(SocBIT_Params params, int itemIndex, RealMatrix rating_errors) {
		
		RealVector itemTopicFeats = params.topicItem.getColumnVector(itemIndex);
		double topicLambda = hypers.topicLambda;
		RealVector topicGrad = itemTopicFeats.mapMultiply(topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int u = 0; u < ds.numUser; u++) {
			double w = params.userDecisionPrefs[u];
			double weighted_rating_err = w * rating_errors.getEntry(u, itemIndex);
			RealVector userTopicFeat = params.topicUser.getColumnVector(u);
			double logisDiff = logisDiff(estimated_ratings.getEntry(u, itemIndex));
			sum = sum.add(userTopicFeat.mapMultiply(weighted_rating_err).mapMultiply(logisDiff));
		}
		
		topicGrad = topicGrad.subtract(sum);
		return topicGrad;
	}

	private RealVector socBIT_userTopicGrad(SocBIT_Params params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {

		RealVector userTopicFeats = params.topicUser.getColumnVector(u);
		double topicLambda = hypers.topicLambda;
		RealVector topicGrad = userTopicFeats.mapMultiply(topicLambda);
		// component wrt rating errors
		RealVector rating_sum = new ArrayRealVector(numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector curItemTopicFeat = params.topicItem.getColumnVector(i);
			double ratingLogisDiff = logisDiff(estimated_ratings.getEntry(u, i));
			RealVector modified_topicFeat = curItemTopicFeat.mapMultiply(rating_errors.getEntry(u, i)).mapMultiply(ratingLogisDiff);
			rating_sum = rating_sum.add(modified_topicFeat);
		}
		// component wrt error of edge weight estimation 
		RealVector edge_weight_sum = new ArrayRealVector(numTopic);
		for (int v = 0; v < ds.numUser; v++) {
			RealVector curUserTopicFeat = params.topicUser.getColumnVector(v);
			double weightLogisDiff = logisDiff(estimated_weights.getEntry(u, v));
			RealVector modified_topicFeat = curUserTopicFeat.mapMultiply(edge_weight_errors.getEntry(u, v)).mapMultiply(weightLogisDiff);
			edge_weight_sum = edge_weight_sum.add(modified_topicFeat);

		}

		double weightLambda = hypers.weightLambda;
		RealVector bigSum = rating_sum.add(edge_weight_sum.mapMultiply(weightLambda));
		topicGrad = topicGrad.subtract(bigSum.mapMultiply(params.userDecisionPrefs[u])); 	// see Eqn. 26
		return topicGrad;
	}
	
	private RealVector ste_itemTopicGrad(Params params, int itemIndex, RealMatrix rating_errors) {
		
		RealVector itemTopicFeats = params.topicItem.getColumnVector(itemIndex);
		RealVector itemTopicGrad = itemTopicFeats.mapMultiply(hypers.topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int u = 0; u < ds.numUser; u++) {
			double rate_err = rating_errors.getEntry(u, itemIndex);
			if (rate_err != 0) {
				double logisDiff = logisDiff(estimated_ratings.getEntry(u, itemIndex));
				RealVector userTopicFeats = params.topicUser.getColumnVector(u);
				RealVector combo_feat = comboFeat(userTopicFeats, u, params);
				
				RealVector correctionByUser = combo_feat.mapMultiply(rate_err).mapMultiply(logisDiff);
				sum = sum.add(correctionByUser);
			}
		}
		
		itemTopicGrad = itemTopicGrad.add(sum);
		return itemTopicGrad;
	}

	private RealVector ste_userTopicGrad(Params params, int u, RealMatrix rating_errors) {
		
		RealVector userTopicGrad = params.topicUser.getColumnVector(u);
		// TODO Auto-generated method stub
		RealVector personal_part = compPersonalPart(u, params, rating_errors);
		
		RealVector friendsPart = new ArrayRealVector(numTopic);
		
		userTopicGrad = personal_part.mapMultiply(alpha).add(friendsPart.mapMultiply(1 - alpha)); 
		return userTopicGrad;
	}

	private RealVector compPersonalPart(int u, Params params,
			RealMatrix rating_errors) {
		RealVector personal_part = new ArrayRealVector(numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector itemTopicFeats = params.topicItem.getColumnVector(i);
			if (rating_errors.getEntry(u, i) > 0) {
				double logisDiff = logisDiff(estimated_ratings.getEntry(u, i));
				double oneRatingErr = rating_errors.getEntry(u, i);
				personal_part = personal_part.add(itemTopicFeats.mapMultiply(oneRatingErr).mapMultiply(logisDiff));
			}
		}
		return personal_part;
	}

	private RealVector comboFeat(RealVector userTopicFeats, int u, Params params) {
		RealVector combo_feat = userTopicFeats.mapMultiply(alpha);
		RealVector friendFeats = new ArrayRealVector(numTopic);
		for (int v = 0; v < ds.numUser; v++) {
			double influenceWeight = ds.edge_weights.getEntry(v, u);
			if (influenceWeight > 0) {
				RealVector vFeat = params.topicUser.getColumnVector(v);
				friendFeats = friendFeats.add(vFeat.mapMultiply(influenceWeight));
			}
		}
		combo_feat = combo_feat.add(friendFeats.mapMultiply(1 - alpha));
		return combo_feat;
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
			double logisDiff = logisDiff(estimated_ratings.getEntry(u, itemIndex));
			sum = sum.add(userBrandFeat.mapMultiply(weighted_rating_err).mapMultiply(logisDiff));
		}
		nextBrandGrad = nextBrandGrad.subtract(sum);
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
			double ratingLogisDiff = logisDiff(estimated_ratings.getEntry(u, i));
			RealVector modified_brandFeat = curItemBrandFeat.mapMultiply(rating_errors.getEntry(u, i)).mapMultiply(ratingLogisDiff);
			rating_sum = rating_sum.add(modified_brandFeat);
		}
		
		// component wrt error of edge weight estimation
		RealVector edge_weight_sum = new ArrayRealVector(ds.numBrand);
		for (int v = 0; v < ds.numUser; v++) {
			RealVector curUserBrandFeat = params.brandUser.getColumnVector(v);
			double weightLogisDiff = logisDiff(estimated_weights.getEntry(u, v));
			RealVector modified_BrandFeat = curUserBrandFeat.mapMultiply(edge_weight_errors.getEntry(u, v)).mapMultiply(weightLogisDiff);
			edge_weight_sum = edge_weight_sum.add(modified_BrandFeat);
		}
		
		double weightLambda = hypers.weightLambda;
		RealVector bigSum = rating_sum.add(edge_weight_sum.mapMultiply(weightLambda));
		nextBrandGrad = nextBrandGrad.subtract(bigSum.mapMultiply(1 - params.userDecisionPrefs[u]));	// see Eqn. 27
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
			double ratingLogisDiff = logisDiff(estimated_ratings.getEntry(u, i));
			rating_sum += rating_errors.getEntry(u, i) * (topicSim - brandSim) * ratingLogisDiff;
		}
		
		double edge_weight_sum = 0;
		for (int v = 0; v < ds.numUser; v++) {
			RealVector theta_v = params.topicUser.getColumnVector(v);
			RealVector beta_v = params.brandUser.getColumnVector(v);
			double topicSim = theta_u.dotProduct(theta_v);
			double brandSim = beta_u.dotProduct(beta_v);
			double weightLogisDiff = logisDiff(estimated_weights.getEntry(u, v));
			edge_weight_sum += edge_weight_errors.getEntry(u, v) * (topicSim - brandSim) * weightLogisDiff;
		}
		
		double weightLambda = hypers.weightLambda;
		double bigSum = rating_sum + weightLambda * edge_weight_sum;
		decisionPrefDiff = decisionPrefDiff - bigSum;
		return decisionPrefDiff;
	}
	
	private double logisDiff(double x) {
		double invExp = Math.exp(-x);
		return invExp/Math.pow(1 + invExp, 2);
	}
	
	/**
	 * NAs in {@link mat} are marked by some invalid value i.e. null, 
	 * in the case of rating, we use -1 as marker 
	 */
	RealMatrix fillNAs(RealMatrix mat, RealMatrix estimated_values) {
		// TODO Auto-generated method stub
		RealMatrix filled_mat = mat;
		for (int i = 0; i < mat.getRowDimension(); i++) {
			for (int j = 0; j < mat.getColumnDimension(); j++) {
				if (filled_mat.getEntry(i, j) == -1) {
					filled_mat.setEntry(i, j, estimated_values.getEntry(i, j));
				}
			}
		}
		return filled_mat;
	}
}
