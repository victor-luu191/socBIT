package core;

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
		// TODO Auto-generated method stub
		return null;
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
