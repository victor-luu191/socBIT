package helpers;

import org.apache.commons.math3.linear.RealVector;

import core.Params;
import core.SocBIT_Params;
import defs.InvalidModelException;
import defs.ParamModelMismatchException;

public class Updater {
	
	private static String model;
	
	public static Params update(Params cParams, double stepSize, Params cGrad, String model) throws ParamModelMismatchException, InvalidModelException {
		
		Params nParams = buildParams(cParams, model);
		
		updateUserComponents(cParams, nParams, cGrad, stepSize);
		updateItemComponents(cParams, nParams, cGrad, stepSize);
		
		return nParams;
	}
	
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

	private static void updateItemComponents(Params cParams, Params nParams, Params cGrad, double stepSize) {
		
		if (model.equalsIgnoreCase("socBIT")) {//cParams instanceof SocBIT_Params && nParams instanceof SocBIT_Params
			updateItemParamsBySocBIT( (SocBIT_Params) cParams, (SocBIT_Params) nParams, (SocBIT_Params) cGrad, stepSize);
		} 
		else {
			if (model.equalsIgnoreCase("STE")) {
				updateItemParamsBySTE(cParams, nParams, cGrad, stepSize);
			}
		}
	}

	private static void updateUserComponents(Params cParams, Params nParams, Params cGrad, double stepSize) {
		
		if (model.equalsIgnoreCase("socBIT")) {
			updateUserParamsBySocBIT((SocBIT_Params) cParams, (SocBIT_Params) nParams, (SocBIT_Params) cGrad, stepSize);
		} 
		else {
			if (model.equalsIgnoreCase("STE")) {
				updateUserParamsBySTE(cParams, nParams, cGrad, stepSize);
			}
		}
	}
	
	private static void updateItemParamsBySTE(Params cParams, Params nParams, Params cGrad, double stepSize) {
		
		int numItem = cParams.topicItem.getColumnDimension();
		for (int i = 0; i < numItem; i++) {
			RealVector curTopicFeat = cParams.topicItem.getColumnVector(i);
			RealVector topicDescent = cGrad.topicItem.getColumnVector(i).mapMultiply(-stepSize);
			RealVector nextTopicFeat = curTopicFeat.add(topicDescent);
			nParams.topicItem.setColumnVector(i, nextTopicFeat);
		}
	}

	private static void updateItemParamsBySocBIT(SocBIT_Params cParams, SocBIT_Params nParams, SocBIT_Params cGrad, double stepSize) {
		
		int numItem = cParams.topicItem.getColumnDimension();
		for (int i = 0; i < numItem; i++) {
			// topic component
			RealVector curTopicFeat = cParams.topicItem.getColumnVector(i);
			RealVector topicDescent = cGrad.topicItem.getColumnVector(i).mapMultiply(-stepSize);
			RealVector nextTopicFeat = curTopicFeat.add(topicDescent);
			nParams.topicItem.setColumnVector(i, nextTopicFeat);
			// brand component
			RealVector curBrandFeat = cParams.brandItem.getColumnVector(i);
			RealVector brandDescent = cGrad.brandItem.getColumnVector(i).mapMultiply(-stepSize);
			RealVector nextBrandFeat = curBrandFeat.add(brandDescent);
			nParams.brandItem.setColumnVector(i, nextBrandFeat);
		}
	}

	private static void updateUserParamsBySTE(Params cParams, Params nParams, Params cGrad, double stepSize) {
		
		int numUser = cParams.topicUser.getColumnDimension();
		for (int u = 0; u < numUser; u++) {
			
			RealVector curTopicFeat = cParams.topicUser.getColumnVector(u);
			RealVector topicDescent = cGrad.topicUser.getColumnVector(u).mapMultiply( -stepSize);
			RealVector nextTopicFeat = curTopicFeat.add(topicDescent);
			nParams.topicUser.setColumnVector(u, nextTopicFeat);
		}
	}

	private static void updateUserParamsBySocBIT(SocBIT_Params cParams, SocBIT_Params nParams, SocBIT_Params cGrad, double stepSize) {
		
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
	}


}
