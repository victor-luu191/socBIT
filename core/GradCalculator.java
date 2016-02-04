package core;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class GradCalculator {
	
	private Parameters params;
	
	// derived fields
	private int numTopic;
	private int numBrand;
	private int numUser;
	private int numItem;
	
	private Estimator estimator;

	private RealMatrix rating_errors;
	private RealMatrix edge_weight_errors;
	
	/**
	 * Construct the calculator of gradients at this set of params. 
	 * The gradients aim to improve estimations to observed ratings and weights 
	 * @param params
	 * @param ratings
	 * @param edge_weights
	 */
	public GradCalculator(Parameters params, RealMatrix ratings, RealMatrix edge_weights) {
		estimator = new Estimator(params);
		RealMatrix estimated_ratings = estimator.estRatings();
		
		RealMatrix estimated_weights = estimator.estWeights();
		// as w_{u, u}'s do NOT exist, we need to exclude errors due to estimating them by the following trick 
		for (int u = 0; u < numUser; u++) {
			edge_weights.setEntry(u, u, estimated_weights.getEntry(u, u));	  
		}
		edge_weight_errors = edge_weights.subtract(estimated_weights);
		
		// XXX: many ratings r_{u,i} are missing as u may not rate i. Again we should exclude the errors from these missing ratings by  
		// similar trick i.e. force the missing ratings equal to estimated values (so that the errors vanish)  
		ratings = fillNAs(ratings, estimated_ratings);
		rating_errors = ratings.subtract(estimated_ratings);
		
		numTopic = params.topicItem.getRowDimension();
		numBrand = params.brandItem.getRowDimension();
		numUser = params.topicUser.getColumnDimension();
		numItem = params.topicItem.getColumnDimension();
	}

	/**
	 * @param itemIndex
	 * @param cParams
	 * @param topicLambda
	 * @param ratingErr: errors of estimating ratings of the item
	 * @return
	 */
	RealVector itemTopicGrad(int itemIndex,  double topicLambda) {
		
		RealVector curTopicGrad = params.topicItem.getRowVector(itemIndex);
		RealVector nextTopicGrad = curTopicGrad.mapMultiply(topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int u = 0; u < numUser; u++) {
			double w = params.userDecisionPrefs[u];
			double weighted_rating_err = w * rating_errors.getEntry(u, itemIndex);
			sum = sum.add(params.topicUser.getColumnVector(u).mapMultiply(weighted_rating_err));
		}
		
		nextTopicGrad = nextTopicGrad.subtract(sum);
		return nextTopicGrad;
	}
	
	RealVector itemBrandGrad(int itemIndex,  double brandLambda) {
		
		RealVector curBrandGrad = params.brandItem.getColumnVector(itemIndex);
		RealVector nextBrandGrad = curBrandGrad.mapMultiply(brandLambda);
		
		RealVector sum = new ArrayRealVector(numBrand);
		for (int u = 0; u < numUser; u++) {
			double w = 1 - params.userDecisionPrefs[u];
			double weighted_rating_err = w * rating_errors.getEntry(u, itemIndex);
			sum = sum.add(params.brandUser.getColumnVector(u).mapMultiply(weighted_rating_err));
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
	RealVector userTopicGrad(int u,  double topicLambda, double weightLambda) {
		RealVector curTopicGrad = params.topicUser.getColumnVector(u);
		RealVector nextTopicGrad = curTopicGrad.mapMultiply(topicLambda);
		// component wrt rating errors
		RealVector rating_sum = new ArrayRealVector(numTopic);
		for (int i = 0; i < numItem; i++) {
			RealVector curItemTopicFeat = params.topicItem.getColumnVector(i);
			RealVector modified_topicFeat = curItemTopicFeat.mapMultiply(rating_errors.getEntry(u, i));
			rating_sum = rating_sum.add(modified_topicFeat);
		}
		// component wrt error of edge weight estimation 
		RealVector edge_weight_sum = new ArrayRealVector(numTopic);
		for (int v = 0; v < numUser; v++) {
			RealVector curUserTopicFeat = params.topicUser.getColumnVector(v);
			RealVector modified_topicFeat = curUserTopicFeat.mapMultiply(edge_weight_errors.getEntry(u, v));
			edge_weight_sum = edge_weight_sum.add(modified_topicFeat);
			
		}
		
		RealVector bigSum = rating_sum.add(edge_weight_sum.mapMultiply(weightLambda));
		nextTopicGrad = nextTopicGrad.subtract(bigSum.mapMultiply(params.userDecisionPrefs[u])); 	// see Eqn. 26
		return nextTopicGrad;
	}
	
	RealVector userBrandGrad(int u,  double brandLambda, double weightLambda) {
		
		RealVector curBrandGrad = params.brandUser.getColumnVector(u);
		RealVector nextBrandGrad = curBrandGrad.mapMultiply(brandLambda);
		// component wrt rating errors
		RealVector rating_sum = new ArrayRealVector(numBrand);
		for (int i = 0; i < numItem; i++) {
			RealVector curItemBrandFeat = params.brandItem.getColumnVector(i);
			RealVector modified_brandFeat = curItemBrandFeat.mapMultiply(rating_errors.getEntry(u, i));
			rating_sum = rating_sum.add(modified_brandFeat);
		}
		
		// component wrt error of edge weight estimation
		RealVector edge_weight_sum = new ArrayRealVector(numBrand);
		for (int v = 0; v < numUser; v++) {
			RealVector curUserBrandFeat = params.brandUser.getColumnVector(v);
			RealVector modified_BrandFeat = curUserBrandFeat.mapMultiply(edge_weight_errors.getEntry(u, v));
			edge_weight_sum = edge_weight_sum.add(modified_BrandFeat);
		}
		
		RealVector bigSum = rating_sum.add(edge_weight_sum.mapMultiply(weightLambda));
		nextBrandGrad = nextBrandGrad.subtract(bigSum.mapMultiply(1 - params.userDecisionPrefs[u]));	// see Eqn. 27
		return nextBrandGrad;
	}
	
	double diffDecisionPref(int u,  double decisionLambda, double weightLambda) {
		
		double userDecisionPref = params.userDecisionPrefs[u];
		double decisionPrefDiff = decisionLambda * (userDecisionPref - 0.5);
		
		RealVector theta_u = params.topicUser.getColumnVector(u);
		RealVector beta_u = params.brandUser.getColumnVector(u);
		
		double rating_sum = 0;
		for (int i = 0; i < numItem; i++) {
			RealVector theta_i = params.topicItem.getColumnVector(i);
			RealVector beta_i = params.brandItem.getColumnVector(i);
			double topicSim = theta_u.dotProduct(theta_i);
			double brandSim = beta_u.dotProduct(beta_i);
			rating_sum += rating_errors.getEntry(u, i) * (topicSim - brandSim);
		}
		
		double edge_weight_sum = 0;
		for (int v = 0; v < numUser; v++) {
			RealVector theta_v = params.topicUser.getColumnVector(v);
			RealVector beta_v = params.brandUser.getColumnVector(v);
			double topicSim = theta_u.dotProduct(theta_v);
			double brandSim = beta_u.dotProduct(beta_v);
			edge_weight_sum += edge_weight_errors.getEntry(u, v) * (topicSim - brandSim);
		}
		
		double bigSum = rating_sum + weightLambda * edge_weight_sum;
		decisionPrefDiff = decisionPrefDiff - bigSum;
		return decisionPrefDiff;
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
