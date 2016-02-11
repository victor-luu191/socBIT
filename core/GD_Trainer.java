package core;

import java.io.IOException;

import myUtil.Savers;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;

public class GD_Trainer {
	
	private static final double EPSILON = 1;
	private static final double ALPHA = 0.5;
	private static final double GAMMA = Math.pow(10, -4);
	private static final double EPSILON_STEP = Math.pow(10, -10);
	
	Dataset ds;
	// settings of this trainer
	int numTopic;
	Hypers hypers;
	private int maxIter;
	private double stepSize;
	
	public GD_Trainer(Dataset ds, int numTopic, Hypers hypers, int maxIter) {
		this.ds = ds;
		this.numTopic = numTopic;
		this.hypers = hypers;
		this.maxIter = maxIter;
		stepSize = 1/ALPHA;
	}
	
	/**
	 * @param initParams
	 * @param resDir
	 * @return local optimal parameters which give a local minimum of the objective function (minimum errors + regularizers)
	 * @throws IOException 
	 */
	Parameters gradDescent(Parameters initParams, String resDir) throws IOException {
		
		System.out.println("Start training...");
		System.out.println("Iter, Objective value");
		
		int numIter = 0;
		Parameters cParams = new Parameters(initParams);
		double cValue = objValue(initParams);
		System.out.println(numIter + ", " + cValue);
		double difference = Double.POSITIVE_INFINITY;
		
//		StringBuilder sbParams = new StringBuilder("iter, ...");
		StringBuilder sbObjValue = new StringBuilder("iter, obj_value \n");
		sbObjValue = sbObjValue.append(numIter + "," + cValue + "\n");
		
		GradCalculator gradCal = new GradCalculator(this);
		// while not convergence and still can try more
		while ( isLarge(difference) && (numIter < maxIter) ) {
			numIter ++;
			Parameters cGrad = gradCal.calGrad(cParams);
			Parameters nParams = lineSearch(cParams, cGrad, cValue);
			double nValue = objValue(nParams);
			sbObjValue = sbObjValue.append(numIter + "," + nValue + "\n");
			difference = nValue - cValue;
			
			// prep for next iter
			cParams = new Parameters(nParams);						
			cValue = nValue;
			System.out.println(numIter + "," + cValue);
		}
		
		if (!isLarge(difference)) {
			System.out.println("Converged to a local minimum :)");
			String fout = resDir + "obj_values.csv";
			Savers.save(sbObjValue.toString(), fout);
			System.out.println("Training done.");
			System.out.println();
		} else {
			System.out.println("Not converged yet but already exceeded the maximum number of iterations. Training stopped!!!");
		}
		
		return cParams;
	}

	private boolean isLarge(double difference) {
		return Math.abs(difference) > EPSILON;
	}
	
	private Parameters lineSearch(Parameters cParams, Parameters cGrad, double cValue) {
		
		Parameters nParams = new Parameters(cParams);
		boolean sufficentReduction = false;
		
		while (!sufficentReduction && (stepSize > EPSILON_STEP)) {
			stepSize = stepSize * ALPHA;
			nParams = update(cParams, stepSize, cGrad);
			// todo: may need some projection here to guarantee some constraints
			double funcDiff = objValue(nParams) - cValue;
			double sqDiff = sqDiff(nParams, cParams);
			double reduction = - GAMMA/stepSize * sqDiff;
			sufficentReduction = (funcDiff < reduction);
			
			if (funcDiff == 0) {
				System.out.println("Meet a local minimum !!!");
				return nParams;
			}
		}
		
		if (sufficentReduction) {
//			System.out.println("Found new params with sufficient reduction");
			return nParams;
		} else {
			System.out.println("Cannot find new params with sufficient reduction. "
					+ "Line search stopped due to step size too small");
			
			return cParams;
		}
	}

	private Parameters update(Parameters cParams, double stepSize, Parameters cGrad) {
		
		Parameters nParams = new Parameters(ds.numUser, ds.numItem, ds.numBrand, this.numTopic);
		for (int u = 0; u < ds.numUser; u++) {
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
		
		for (int i = 0; i < ds.numItem; i++) {
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
		
		return nParams;
	}

	private double objValue(Parameters params) {
		
		Estimator estimator = new Estimator(params);
		RealMatrix estimated_ratings = estimator.estRatings();
		RealMatrix estimated_weights = estimator.estWeights();
		
		RealMatrix edge_weight_errors = ErrorCal.edgeWeightErrors(estimated_weights, ds.edge_weights);
		RealMatrix rating_errors = ErrorCal.ratingErrors(estimated_ratings, ds.ratings);
		
		double val = square(rating_errors.getFrobeniusNorm());
		val += hypers.weightLambda * square(edge_weight_errors.getFrobeniusNorm());
		val += hypers.topicLambda * ( square(params.topicUser.getFrobeniusNorm()) + square(params.topicItem.getFrobeniusNorm()) );
		val += hypers.brandLambda * ( square(params.brandUser.getFrobeniusNorm()) + square(params.brandItem.getFrobeniusNorm()) );
		for (int u = 0; u < ds.numUser; u++) {
			val += hypers.decisionLambda * square(params.userDecisionPrefs[u] - 0.5);
		}
		
		return val;
	}
	
	private double sqDiff(Parameters p1, Parameters p2) {

		double sq_diff = square(p1.topicUser.subtract(p2.topicUser).getFrobeniusNorm());
		sq_diff += square(p1.topicItem.subtract(p2.topicItem).getFrobeniusNorm());
		sq_diff += square(p1.brandUser.subtract(p2.brandUser).getFrobeniusNorm());
		sq_diff += square(p1.brandItem.subtract(p2.brandItem).getFrobeniusNorm());
		for (int u = 0; u < ds.numUser; u++) {
			sq_diff += square(p1.userDecisionPrefs[u] - p2.userDecisionPrefs[u]);
		}

		return sq_diff;
	}
	
	private double square(double d) {
		return Math.pow(d, 2);
	}
}
