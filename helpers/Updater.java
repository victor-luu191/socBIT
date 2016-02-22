package helpers;

import org.apache.commons.math3.linear.RealVector;

import core.Params;
import core.SocBIT_Params;
import defs.InvalidModelException;
import defs.ParamModelMismatchException;

public class Updater {
	
	public static Params update(Params cParams, double stepSize, Params cGrad, String model) throws ParamModelMismatchException, InvalidModelException {
		
//		Params nParams = buildParams(cParams, model);
		
		Params nParams = updateItemComponents(cParams, cGrad, stepSize, model);
		nParams = updateUserComponents(nParams,  cGrad, stepSize, model);
		return nParams;
	}
	
	private static Params updateItemComponents(Params cParams, Params cGrad, double stepSize, String model) {
		
//		int numUser = getNumUser(cParams);	int numItem = getNumItem(cParams);	int numTopic = getNumTopic(cParams);
//		Params nParams = new Params(numUser, numItem, numTopic);
		
		Params nParams = new Params(cParams);
		if (model.equalsIgnoreCase("socBIT")) {//cParams instanceof SocBIT_Params && nParams instanceof SocBIT_Params
			nParams = updateItemParamsBySocBIT( (SocBIT_Params) cParams, (SocBIT_Params) cGrad, stepSize);
		} 
		else {
			if (model.equalsIgnoreCase("STE")) {
				nParams = updateItemParamsBySTE(cParams, cGrad, stepSize);
			}
		}
		return nParams;
	}

	private static int getNumTopic(Params params) {
		return params.topicItem.getRowDimension();
	}

	private static int getNumItem(Params params) {
		return params.topicItem.getColumnDimension();
	}

	private static int getNumUser(Params params) {
		return params.topicUser.getColumnDimension();
	}

	private static Params updateUserComponents(Params cParams,  Params cGrad, double stepSize, String model) {
		
		Params nParams = new Params(cParams);
		if (model.equalsIgnoreCase("socBIT")) {
			nParams = updateUserParamsBySocBIT((SocBIT_Params) cParams,  (SocBIT_Params) cGrad, stepSize);
		} 
		else {
			if (model.equalsIgnoreCase("STE")) {
				nParams = updateUserParamsBySTE(cParams,  cGrad, stepSize);
			}
		}
		return nParams;
	}
	
	private static Params updateItemParamsBySTE(Params params, Params cGrad, double stepSize) {
		
		Params nParams = new Params(params);
		int numItem = params.topicItem.getColumnDimension();
		for (int i = 0; i < numItem; i++) {
			RealVector curTopicFeat = params.topicItem.getColumnVector(i);
			RealVector topicDescent = cGrad.topicItem.getColumnVector(i).mapMultiply(-stepSize);
			RealVector nextTopicFeat = curTopicFeat.add(topicDescent);
			nParams.topicItem.setColumnVector(i, nextTopicFeat);
		}
		return nParams;
	}

	private static SocBIT_Params updateItemParamsBySocBIT(SocBIT_Params params, SocBIT_Params cGrad, double stepSize) {
		
		SocBIT_Params nParams = new SocBIT_Params(params);
		int numItem = params.topicItem.getColumnDimension();
		for (int i = 0; i < numItem; i++) {
			// topic component
			RealVector curTopicFeat = params.topicItem.getColumnVector(i);
			RealVector topicDescent = cGrad.topicItem.getColumnVector(i).mapMultiply(-stepSize);
			RealVector nextTopicFeat = curTopicFeat.add(topicDescent);
			nParams.topicItem.setColumnVector(i, nextTopicFeat);
			// brand component
			RealVector curBrandFeat = params.brandItem.getColumnVector(i);
			RealVector brandDescent = cGrad.brandItem.getColumnVector(i).mapMultiply(-stepSize);
			RealVector nextBrandFeat = curBrandFeat.add(brandDescent);
			nParams.brandItem.setColumnVector(i, nextBrandFeat);
		}
		return nParams;
	}

	private static Params updateUserParamsBySTE(Params cParams,  Params cGrad, double stepSize) {
		
		Params nParams = new Params(cParams);
		int numUser = getNumUser(cParams);
		for (int u = 0; u < numUser; u++) {
			
			RealVector curTopicFeat = cParams.topicUser.getColumnVector(u);
			RealVector topicDescent = cGrad.topicUser.getColumnVector(u).mapMultiply( -stepSize);
			RealVector nextTopicFeat = curTopicFeat.add(topicDescent);
			nParams.topicUser.setColumnVector(u, nextTopicFeat);
		}
		return nParams;
	}

	private static SocBIT_Params updateUserParamsBySocBIT(SocBIT_Params cParams,  SocBIT_Params cGrad, double stepSize) {
		
		SocBIT_Params nParams = new SocBIT_Params(cParams);
		int numUser = cParams.topicUser.getColumnDimension();
		for (int u = 0; u < numUser; u++) {
			// user decision pref
			nParams.userDecisionPrefs[u] = cParams.userDecisionPrefs[u] - stepSize * cGrad.userDecisionPrefs[u];
			// topic component
			RealVector curTopicFeat = cParams.topicUser.getColumnVector(u);
			RealVector topicDescent = cGrad.topicUser.getColumnVector(u).mapMultiply( -stepSize);
			RealVector nextTopicFeat = curTopicFeat.add(topicDescent);
			nParams.topicUser.setColumnVector(u, nextTopicFeat);
			// brand component
			RealVector curBrandFeat = cParams.brandUser.getColumnVector(u);
			RealVector brandDescent = cGrad.brandUser.getColumnVector(u).mapMultiply(-stepSize);
			RealVector nextBrandFeat = curBrandFeat.add(brandDescent);
			nParams.brandUser.setColumnVector(u, nextBrandFeat);
		}
		return nParams;
	}

	@SuppressWarnings("unused")
	private static Params buildParams(Params cParams, String model) throws ParamModelMismatchException, InvalidModelException {
		
		if (model.equalsIgnoreCase("STE")) {
			return cParams;
		} 
		else {
			if (model.equalsIgnoreCase("socBIT")) {
				if (cParams instanceof SocBIT_Params) {
					return (SocBIT_Params) cParams;
				} 
				else {
					String msg = "Input params type and model mismatch!!!";
					throw new ParamModelMismatchException(msg);
				}
			} 
			else {
				throw new InvalidModelException();
			}
		}
	}
}
