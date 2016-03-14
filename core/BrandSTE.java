package core;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;
import defs.Params;
import defs.SocBIT_Params;

class BrandSTE extends STE_Cal {
	
	Dataset ds; 
	Hypers hypers;
	
	public BrandSTE(Dataset ds, Hypers hypers) {
		super(ds, hypers);
	}

	@Override
	double objValue(Params params) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	RealMatrix estRatings(Params params) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	RealMatrix calRatingErrors(Params params) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	double estOneRating(int u, int i, Params params) {
		// TODO Auto-generated method stub
		SocBIT_Params castParams = (SocBIT_Params) params;
		double topicRating = super.estOneRating(u, i, params);
		double decPref = castParams.userDecisionPrefs[u];
		double gamma = decPref * topicRating;
		double brandRating = calBrandRating(u,i, castParams);
		gamma += (1 - decPref)*brandRating;
		return gamma;
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
