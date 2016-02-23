package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class STE_GradCal extends GradCal {

	// this alpha is redundant (it is already included in hypers), but for later brevity, we allow this redundancy
	// for the meaning of alpha, see in hypers
	private double alpha;
	
	public STE_GradCal(Trainer trainer) {
		super(trainer);
		// TODO Auto-generated constructor stub
		this.alpha = hypers.alpha;
	}

	@Override
	Params calculate(Params params) {
		
		STE_Cal ste_estimator = new STE_Cal(ds, hypers);
		estimated_ratings = ste_estimator.estRatings(params);
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
	
	@Override
	RealVector itemTopicGrad(Params params, int itemIndex, RealMatrix rating_errors) {
		
		RealVector itemTopicFeats = params.topicItem.getColumnVector(itemIndex);
		RealVector itemTopicGrad = itemTopicFeats.mapMultiply(hypers.topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int u = 0; u < ds.numUser; u++) {
			double rate_err = rating_errors.getEntry(u, itemIndex);
			if (rate_err != 0) {
				double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, itemIndex));
				RealVector userTopicFeats = params.topicUser.getColumnVector(u);
				RealVector combo_feat = comboFeat(userTopicFeats, u, params);
				
				RealVector correctionByUser = combo_feat.mapMultiply(rate_err).mapMultiply(logisDiff);
				sum = sum.add(correctionByUser);
			}
		}
		
		itemTopicGrad = itemTopicGrad.add(sum);
		return itemTopicGrad;
	}

	private RealVector userTopicGrad(Params params, int u, RealMatrix rating_errors) {
		
		RealVector userTopicFeats = params.topicUser.getColumnVector(u);
		RealVector userTopicGrad = userTopicFeats.mapMultiply(hypers.topicLambda);
		
		RealVector personal_part = compPersonalPart(u, params, rating_errors);
		RealVector influenceePart = compInfluenceePart(u, params, rating_errors);
		
		RealVector sum = personal_part.mapMultiply(alpha).add(influenceePart.mapMultiply(1 - alpha));
		userTopicGrad = userTopicGrad.add(sum); 
		return userTopicGrad;
	}
	
	private RealVector compInfluenceePart(int u, Params params, RealMatrix rating_errors) {
		// influencee: those who are influenced by/trust u, thus include u's feat in their rating
		RealVector influenceePart = new ArrayRealVector(numTopic);	
		for (int v = 0; v < ds.numUser; v++) {
			double influencedLevel = ds.edge_weights.getEntry(u, v);
			if (influencedLevel > 0) {
				for (int i = 0; i < ds.numItem; i++) {
					double oneRatingErr = rating_errors.getEntry(v, i);
					if (oneRatingErr > 0) {
						RealVector itemTopicFeats = params.topicItem.getColumnVector(i);
						double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(v, i));
						double weight = influencedLevel * oneRatingErr * logisDiff;
						influenceePart = influenceePart.add(itemTopicFeats.mapMultiply(weight));
					}
				}
			}
		}
		return influenceePart;
	}

	private RealVector compPersonalPart(int u, Params params, RealMatrix rating_errors) {
		
		RealVector personal_part = new ArrayRealVector(numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector itemTopicFeats = params.topicItem.getColumnVector(i);
			if (rating_errors.getEntry(u, i) > 0) {
				double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
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
}
