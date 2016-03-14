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
		return super.calItemTopicGrad(params, itemIndex);
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
				sum = sum.add(correctionByUser);
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
		return super.calUserTopicGrad(params, u);
	}
	
	private RealVector calUserBrandGrad(SocBIT_Params params, int u) {
		// TODO Auto-generated method stub
		return null;
	}
}
