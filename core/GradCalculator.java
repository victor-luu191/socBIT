package core;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Hypers;

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
		
		edge_weight_errors = edge_weights.subtract(estimated_weights);
		// XXX: some ratings r_{u,i} are NAs as u may not rate i. 
		// Thus, 1st thing to do is to fill in the ratings by estimated values  
		ratings = fillNAs(ratings, estimated_ratings);
		rating_errors = ratings.subtract(estimated_ratings);
		
		numTopic = params.topicItem.getRowDimension();
		numBrand = params.brandItem.getRowDimension();
		numUser = params.topicUser.getColumnDimension();
		numItem = params.topicItem.getColumnDimension();
	}
	
	
	private RealMatrix fillNAs(RealMatrix ratings, RealMatrix estimated_ratings) {
		// TODO Auto-generated method stub
		return null;
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
	
	RealVector userBrandGrad(int u,  double brandLambda, RealVector ratingErr, RealVector strengthErr, double weightLambda) {
		
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
	
	double diffDecisionPref(int userIndex,  Hypers hypers, RealVector ratingErr, RealVector strengthErr) {
		
		double decisionPrefDiff = 0;
		// TODO
		return decisionPrefDiff;
	}
}
