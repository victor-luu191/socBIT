package core;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;

public class GradCalculator {
	
	private Hypers hypers;
	// derived fields
	private int numTopic;
	Dataset ds;
	private RealMatrix estimated_ratings;
	private RealMatrix estimated_weights;
	
	public GradCalculator(GD_Trainer trainer) {
		
		numTopic = trainer.numTopic;
		ds = trainer.ds;
		hypers = trainer.hypers;
	}
	
	/**
	 * Main work is here !!!
	 * Compute the gradient at a given set of params, by computing all the components of the gradient
	 * @param params
	 * @return the complete gradient with all its components
	 */
	Parameters calGrad(Parameters params) {
		
		Estimator estimator = new Estimator(params);
		estimated_ratings = estimator.estRatings();
		RealMatrix bounded_ratings = estimator.bound(estimated_ratings);
		
		estimated_weights = estimator.estWeights();
		RealMatrix bounded_weights = estimator.bound(estimated_weights);
		
		RealMatrix edge_weight_errors = ErrorCal.edgeWeightErrors(bounded_weights, ds.edge_weights);	// estimated_weights
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);					// estimated_ratings
		
		Parameters grad = new Parameters(ds.numUser, ds.numItem, ds.numBrand, this.numTopic);
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

	/**
	 * @param itemIndex
	 * @param cParams
	 * @param topicLambda
	 * @param ratingErr: errors of estimating ratings of the item
	 * @return
	 */
	RealVector itemTopicGrad(Parameters params, int itemIndex, RealMatrix rating_errors) {
		
		RealVector curTopicGrad = params.topicItem.getColumnVector(itemIndex);
		double topicLambda = hypers.topicLambda;
		RealVector nextTopicGrad = curTopicGrad.mapMultiply(topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int u = 0; u < ds.numUser; u++) {
			double w = params.userDecisionPrefs[u];
			double weighted_rating_err = w * rating_errors.getEntry(u, itemIndex);
			RealVector userTopicFeat = params.topicUser.getColumnVector(u);
			double logisDiff = logisDiff(estimated_ratings.getEntry(u, itemIndex));
			sum = sum.add(userTopicFeat.mapMultiply(weighted_rating_err).mapMultiply(logisDiff));
		}
		
		nextTopicGrad = nextTopicGrad.subtract(sum);
		return nextTopicGrad;
	}

	RealVector itemBrandGrad(Parameters params, int itemIndex, RealMatrix rating_errors) {
		
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
	
	/**
	 * 
	 * @param u
	 * @param cParams
	 * @param topicLambda
	 * @param ratingErr
	 * @param weightLambda 
	 * @param strengthErr: errors in estimating strength of relationships of the user 
	 * @return
	 */
	RealVector userTopicGrad(Parameters params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {
		
		RealVector curTopicGrad = params.topicUser.getColumnVector(u);
		double topicLambda = hypers.topicLambda;
		RealVector nextTopicGrad = curTopicGrad.mapMultiply(topicLambda);
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
		nextTopicGrad = nextTopicGrad.subtract(bigSum.mapMultiply(params.userDecisionPrefs[u])); 	// see Eqn. 26
		return nextTopicGrad;
	}
	
	RealVector userBrandGrad(Parameters params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {
		
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
	
	double userDecisionPrefDiff(Parameters params, int u, RealMatrix rating_errors, RealMatrix edge_weight_errors) {
		
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
