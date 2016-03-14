package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import defs.Params;
import defs.SocBIT_Params;

class BrandSTE_GradCal extends STE_GradCal {

	
	public BrandSTE_GradCal(Trainer trainer) {
		super(trainer);
		calculator = new BrandSTE_Cal(ds, hypers);
	}

	@Override
	Params calculate(Params params) {
		
		SocBIT_Params grad = new SocBIT_Params(ds.numUser, ds.numItem, ds.numBrand, numTopic);
		SocBIT_Params castParams = (SocBIT_Params) params;
		calculator.estRatings(params);
		calculator.calRatingErrors(params);
		
		// gradients for user feats
		for (int u = 0; u < ds.numUser; u++) {
			RealVector userTopicGrad = calUserTopicGrad(params, u);
			grad.topicUser.setColumnVector(u, userTopicGrad);
			RealVector userBrandGrad = calUserBrandGrad(castParams, u);
			grad.brandUser.setColumnVector(u, userBrandGrad);
			
			grad.userDecisionPrefs[u] = userDecPrefDiff(castParams, u);
		}
		// gradients for item feats
		for (int i = 0; i < ds.numItem; i++) {
			grad.topicItem.setColumnVector(i, calItemTopicGrad(params, i));
			RealVector itemBrandGrad = calItemBrandGrad(castParams, i);
			grad.brandItem.setColumnVector(i, itemBrandGrad);
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
				SocBIT_Params castParams = (SocBIT_Params) params;
				double decPref = castParams.userDecisionPrefs[u];
				sum = sum.add(correctionByUser.mapMultiply(decPref));
			}
		}
		
		itemTopicGrad = itemTopicGrad.add(sum);
		return itemTopicGrad;
	}
	
	private RealVector calItemBrandGrad(SocBIT_Params params, int i) {
		
		RealVector itemBrandFeats = params.brandItem.getColumnVector(i);
		RealVector itemBrandGrad = itemBrandFeats.mapMultiply(hypers.brandLambda);
		
		RealVector sum = new ArrayRealVector(ds.numBrand);
		for (int u = 0; u < ds.numUser; u++) {
			double rate_err = rating_errors.getEntry(u, i);
			if (rate_err != 0) {
				double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
				RealVector personalBrandFeats = params.brandUser.getColumnVector(u);
				RealVector comboBrandFeat = calComboBrandFeat(personalBrandFeats, u, params);
				RealVector correctionByUser = comboBrandFeat.mapMultiply(rate_err).mapMultiply(logisDiff);
				
				double decPref = params.userDecisionPrefs[u];
				sum = sum.add(correctionByUser.mapMultiply(1 - decPref));
			}
		}
		itemBrandGrad = itemBrandGrad.add(sum);
		return itemBrandGrad;
	}

	private RealVector calComboBrandFeat(RealVector personalBrandFeats, int u, SocBIT_Params params) {
		
		RealVector friendFeats = new ArrayRealVector(ds.numBrand);
		for (int v = 0; v < ds.numUser; v++) {
			double influenceWeight = ds.edge_weights.getEntry(v, u);
			if (influenceWeight > 0) {
				RealVector vFeat = params.brandUser.getColumnVector(v);
				friendFeats = friendFeats.add(vFeat.mapMultiply(influenceWeight));
			}
		}
		RealVector comboFeat = personalBrandFeats.mapMultiply(hypers.alpha);
		comboFeat = comboFeat.add(friendFeats.mapMultiply(1 - hypers.alpha));
		return comboFeat;
	}

	private double userDecPrefDiff(SocBIT_Params params, int u) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	RealVector calUserTopicGrad(Params params, int u) {
		// TODO Auto-generated method stub
		RealVector userTopicFeats = params.topicUser.getColumnVector(u);
		RealVector userTopicGrad = userTopicFeats.mapMultiply(hypers.topicLambda);
		
		SocBIT_Params castParams = (SocBIT_Params) params;
		RealVector personal_part = compPersonalPart(u, castParams);
		RealVector influenceePart = compInfluenceePart(u, castParams);
		
		RealVector sum = personal_part.mapMultiply(alpha).add(influenceePart.mapMultiply(1 - alpha));
		userTopicGrad = userTopicGrad.add(sum); 
		return userTopicGrad;
	}
	
	private RealVector calUserBrandGrad(SocBIT_Params params, int u) {
		// TODO Auto-generated method stub
		return null;
	}

	
	protected RealVector compPersonalPart(int u, SocBIT_Params params) {
		
		RealVector personal_part = new ArrayRealVector(numTopic);
		for (int i = 0; i < ds.numItem; i++) {
			RealVector itemTopicFeats = params.topicItem.getColumnVector(i);
			if (rating_errors.getEntry(u, i) > 0) {
				double logisDiff = UtilFuncs.logisDiff(estimated_ratings.getEntry(u, i));
				double oneRatingErr = rating_errors.getEntry(u, i);
				double personalDecPref = params.userDecisionPrefs[u];
				double coef = oneRatingErr*logisDiff*personalDecPref;
				personal_part = personal_part.add(itemTopicFeats.mapMultiply(coef));
			}
		}
		return personal_part;
	}
	
	protected RealVector compInfluenceePart(int u, SocBIT_Params params) {
		
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
						double vDecPref = params.userDecisionPrefs[v];
						double coef = influencedLevel * oneRatingErr * logisDiff * vDecPref;
						influenceePart = influenceePart.add(itemTopicFeats.mapMultiply(coef));
						
					}
				}
			}
		}
		return influenceePart;
	}
}
