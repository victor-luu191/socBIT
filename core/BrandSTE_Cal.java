package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;
import defs.Params;
import defs.SocBIT_Params;

class BrandSTE_Cal extends STE_Cal {
	
	public BrandSTE_Cal(Dataset ds, Hypers hypers) {
		super(ds, hypers);
	}
	
	@Override
	double regularization(Params params) {
		
		double topicPart = super.regularization(params);
		double brandPart = 0;
		double decPart = 0;
		if (params instanceof SocBIT_Params) {
			SocBIT_Params castParams = (SocBIT_Params) params;
			brandPart = regBrandFeats(castParams);
			decPart = regDecPref(castParams);
		}
		return topicPart + brandPart + decPart;
	}

	private double regDecPref(SocBIT_Params params) {
		
//		SocBIT_Params castParams = (SocBIT_Params) params;
		double sum = 0;
		for (int u = 0; u < ds.numUser; u++) {
			double decPref = params.userDecisionPrefs[u];
			sum += UtilFuncs.square(decPref - 0.5);
		}
		return hypers.decisionLambda * sum;
	}

	private double regBrandFeats(SocBIT_Params params) {
		double userBrandFeatNorm = params.brandUser.getFrobeniusNorm();
		double itemBrandFeatNorm = params.brandItem.getFrobeniusNorm();
		double brandPart = hypers.brandLambda * (UtilFuncs.square(userBrandFeatNorm) + UtilFuncs.square(itemBrandFeatNorm));
		return brandPart;
	}
	
	@Override
	double estOneRating(int u, int i, Params params) {
		double topicRating = super.estOneRating(u, i, params);
		
		SocBIT_Params castParams = (SocBIT_Params) params;
		double decPref = castParams.userDecisionPrefs[u];
		double brandRating = calBrandRating(u,i, castParams);
		
		return decPref*topicRating + (1 - decPref)*brandRating;
	}

	private double calBrandRating(int u, int i, SocBIT_Params params) {
		
		RealVector itemBrandFeats = params.brandItem.getColumnVector(i);
		RealVector userBrandFeats = params.brandUser.getColumnVector(u);
		double personal = userBrandFeats.dotProduct(itemBrandFeats);
		double social = 0;
		// actually we only loop thru the set of friends of u, 
		// thus, later we can speed up by caching the set of friends of each user
		for (int v = 0; v < ds.numUser; v++) { 
			double influenceWeight = ds.edge_weights.getEntry(v, u);
			if (influenceWeight > 0) {
				RealVector vBrandFeats = params.brandUser.getColumnVector(v);
				social += influenceWeight*vBrandFeats.dotProduct(itemBrandFeats);
			}
		}
				
		double alpha = hypers.alpha;
		return alpha*personal + (1 - alpha)*social;
	}
	
	
}
