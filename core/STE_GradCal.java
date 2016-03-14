package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import defs.Params;

public class STE_GradCal extends GradCal {

	// this alpha is redundant (it is already included in hypers), but for later brevity, we allow this redundancy
	// for the meaning of alpha, see in hypers
	protected double alpha;
	protected STE_Cal calculator;
	
	public STE_GradCal(Trainer trainer) {
		
		numTopic = trainer.numTopic;
		ds = trainer.ds;
		hypers = trainer.hypers;
		this.alpha = hypers.alpha;
		calculator = new STE_Cal(ds, hypers);
	}

	@Override
	Params calculate(Params params) {
		
		calculator.estRatings(params);	// for each set of params, need to re-estimate ratings
		calculator.calRatingErrors(params);
		
		Params grad = new Params(ds.numUser, ds.numItem, numTopic);
		// gradients for users
		for (int u = 0; u < ds.numUser; u++) {
			RealVector userTopicGrad = calUserTopicGrad(params, u);
			grad.topicUser.setColumnVector(u, userTopicGrad);
		}
		// gradients for items
		for (int i = 0; i < ds.numItem; i++) {
			grad.topicItem.setColumnVector(i, calItemTopicGrad(params, i));
		}
		return grad;
	}
	
	@Override
	RealVector calItemTopicGrad(Params params, int itemIndex) {
		
		RealVector itemTopicFeats = params.topicItem.getColumnVector(itemIndex);
		RealVector itemTopicGrad = itemTopicFeats.mapMultiply(hypers.topicLambda);
		
		RealVector sum = new ArrayRealVector(numTopic);
		for (int u = 0; u < ds.numUser; u++) {
			double rate_err = rating_errors.getEntry(u, itemIndex);
			if (rate_err != 0) {
				double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, itemIndex));
				RealVector userTopicFeats = params.topicUser.getColumnVector(u);
				RealVector comboTopicFeat = comboTopicFeat(userTopicFeats, u, params);
				
				RealVector correctionByUser = comboTopicFeat.mapMultiply(rate_err).mapMultiply(logisDiff);
				sum = sum.add(correctionByUser);
			}
		}
		
		itemTopicGrad = itemTopicGrad.add(sum);
		return itemTopicGrad;
	}

	RealVector calUserTopicGrad(Params params, int u) {
		
		RealVector userTopicFeats = params.topicUser.getColumnVector(u);
		RealVector userTopicGrad = userTopicFeats.mapMultiply(hypers.topicLambda);
		
		RealVector personal_part = compPersonalPart(u, params);
		RealVector influenceePart = compInfluenceePart(u, params);
		
		RealVector sum = personal_part.mapMultiply(alpha).add(influenceePart.mapMultiply(1 - alpha));
		userTopicGrad = userTopicGrad.add(sum); 
		return userTopicGrad;
	}
	
	protected RealVector compInfluenceePart(int u, Params params) {
		
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

	protected RealVector compPersonalPart(int u, Params params) {
		
		RealVector personal_part = new ArrayRealVector(numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector itemTopicFeats = params.topicItem.getColumnVector(i);
			if (rating_errors.getEntry(u, i) > 0) {
				double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
				double oneRatingErr = rating_errors.getEntry(u, i);
				personal_part = personal_part.add(itemTopicFeats.mapMultiply(oneRatingErr*logisDiff));
			}
		}
		return personal_part;
	}

	protected RealVector comboTopicFeat(RealVector userTopicFeats, int u, Params params) {
		
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
