package core;

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
	
	/**
	 * 
	 * @param initParams
	 * @param resDir
	 * @return local optimal parameters which give a local minimum of the objective function (minimum errors + regularizers)
	 */
	Parameters gradDescent(Parameters initParams, String resDir) {
		
		// TODO Auto-generated method stub
		Parameters cParams = new Parameters(initParams);
		double cValue = objValue(initParams);
		double difference = Double.POSITIVE_INFINITY;
		int numIter = 0;
		
		GradCalculator gradCal = new GradCalculator(this);
		// while not convergence and still can try more
		while ( (Math.abs(difference) > EPSILON) && (numIter < maxIter) ) {
			numIter ++;
			Parameters cGrad = gradCal.calGrad(cParams);
			Parameters nParams = lineSearch(cParams, cGrad, cValue);
			double nValue = objValue(nParams);
			
			difference = nValue - cValue;
			// prep for next iter
			cParams = new Parameters(nParams);						
			cValue = nValue;
		}
		
		return cParams;
	}
	
	private Parameters lineSearch(Parameters cParams, Parameters cGrad, double cValue) {
		
		stepSize = 1/ALPHA;
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
				System.out.println("meet a local minimum !!!");
				return nParams;
			}
		}
		
		if (sufficentReduction) {
			System.out.println("found new params with sufficient reduction");
			return nParams;
		} else {
			System.out.println("Cannot find new params with sufficient reduction. "
					+ "Line search stopped due to step size too small");
			return cParams;
		}
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

	private Parameters update(Parameters cParams, double stepSize, Parameters cGrad) {
		// TODO Auto-generated method stub
		Parameters nParams = new Parameters(ds.numUser, ds.numItem, numTopic, ds.numBrand);
		for (int u = 0; u < ds.numUser; u++) {
			RealVector curTopicFeat = cParams.topicUser.getColumnVector(u);
			RealVector descentGrad = cGrad.topicUser.getColumnVector(u).mapMultiply(-stepSize);
			RealVector nextTopicFeat = curTopicFeat.add(descentGrad);
			nParams.topicUser.setColumnVector(u, nextTopicFeat);
		}
		
		return null;
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
	
	double square(double d) {
		return Math.pow(d, 2);
	}
}
