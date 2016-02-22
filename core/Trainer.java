package core;

import helpers.Updater;
import helpers.UtilFuncs;

import java.io.IOException;

import myUtil.Savers;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import defs.Dataset;
import defs.Hypers;
import defs.InvalidModelException;
import defs.ParamModelMismatchException;

public class Trainer {
	
	private static final double EPSILON = 1;
	private static final double INVERSE_STEP = 0.5;
	private static final double GAMMA = Math.pow(10, -4);
	private static final double EPSILON_STEP = Math.pow(10, -10);
	
	Dataset ds;
	
	// settings of this trainer
	String model;
	int numTopic;
	Hypers hypers;
	private int maxIter;
	private double stepSize;
	
	public Trainer(String model, Dataset ds, int numTopic, Hypers hypers, int maxIter) {
		this.model = model;
		this.ds = ds;
		this.numTopic = numTopic;
		this.hypers = hypers;
		this.maxIter = maxIter;
		stepSize = 1/INVERSE_STEP;
	}
	
	/**
	 * @param initParams
	 * @param resDir
	 * @return local optimal parameters which give a local minimum of the objective function (minimum errors + regularizers)
	 * @throws IOException, InvalidModelException and  ParamModelMismatchException
	 */
	Params gradDescent(Params initParams, String resDir) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		printStartMsg();
		int numIter = 0;
//		StringBuilder sbParams = new StringBuilder("iter, ...");
		StringBuilder sbObjValue = new StringBuilder("iter, obj_value \n");
		
		Params cParams = buildParams(initParams, model);
		double cValue = objValue(initParams);
		sbObjValue = sbObjValue.append(numIter + "," + cValue + "\n");
		System.out.println(numIter + ", " + cValue);
		double difference = Double.POSITIVE_INFINITY;
		
		GradCal gradCal = new GradCal(this);
		// while not convergence and still can try more
		while ( isLarge(difference) && (numIter < maxIter) ) {
			numIter ++;
			Params cGrad = gradCal.calculate(cParams, this.model);
			Params nParams = lineSearch(cParams, cGrad, cValue);
			double nValue = objValue(nParams);
			sbObjValue = sbObjValue.append(numIter + "," + nValue + "\n");
			difference = nValue - cValue;
			
			// prep for next iter
			cParams = buildParams(nParams, model);						
			cValue = nValue;
			System.out.println(numIter + "," + cValue);
		}
		
		if (!isLarge(difference)) {
			printConvergeMsg();
			String fout = resDir + "obj_values.csv";
			Savers.save(sbObjValue.toString(), fout);
		} else {
			System.out.println("Not converged yet but already exceeded the maximum number of iterations. Training stopped!!!");
		}
		
		return cParams;
	}

	private void printStartMsg() {
		System.out.println("Start training...");
		System.out.println("Iter, Objective value");
	}

	private void printConvergeMsg() {
		System.out.println("Converged to a local minimum :)");
		System.out.println("Training done.");
		System.out.println();
	}

	private boolean isLarge(double difference) {
		return Math.abs(difference) > EPSILON;
	}
	
	private Params lineSearch(Params cParams, Params cGrad, double cValue) throws ParamModelMismatchException, InvalidModelException {
		
		Params nParams = buildParams(cParams, this.model);
		boolean sufficentReduction = false;
		
		while (!sufficentReduction && (stepSize > EPSILON_STEP)) {
			stepSize = stepSize * INVERSE_STEP;
			nParams = Updater.update(cParams, stepSize, cGrad, this.model);
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
			System.out.println("Cannot find better new params  (i.e. with sufficient reduction). "
								+ "Line search stopped due to step size too small");
			
			return cParams;
		}
	}

	private Params buildParams(Params cParams, String model) throws ParamModelMismatchException, InvalidModelException {
		
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

	private double objValue(Params params) throws InvalidModelException {
		
		double val ;
		if (model.equalsIgnoreCase("socBIT")) {
			SocBIT_Params castParams = (SocBIT_Params) params;
			val = castParams.objValue(ds, hypers);
		} 
		else {
			if (model.equalsIgnoreCase("STE")) {
				val = valueBySTE(params);
			} else {
				throw new InvalidModelException();
			}
		}
		
		return val;
	}

	private double valueBySTE(Params params) {
		
		double userFeatsNorm = params.topicUser.getFrobeniusNorm();
		double itemFeatsNorm = params.topicItem.getFrobeniusNorm();
		double val = hypers.topicLambda * (square(userFeatsNorm) + square(itemFeatsNorm));	// regularized part
		
		RealMatrix rating_errors = ste_ratingErrors(params);
		val += square(rating_errors.getFrobeniusNorm());;
		return val;
	}

	private RealMatrix ste_ratingErrors(Params params) {
		
		STE_estimator ste_estimator = new STE_estimator(params, hypers.alpha, ds.edge_weights);
		RealMatrix estimated_ratings = ste_estimator.estRatings();
		RealMatrix bounded_ratings = UtilFuncs.bound(estimated_ratings);
		RealMatrix rating_errors = ErrorCal.ratingErrors(bounded_ratings, ds.ratings);
		return rating_errors;
	}

	private double sqDiff(Params p1, Params p2) throws InvalidModelException {

		if (model.equalsIgnoreCase("socBIT")) {
			SocBIT_Params cast_p1 = (SocBIT_Params) p1;
			SocBIT_Params cast_p2 = (SocBIT_Params) p2;
			return cast_p1.sqDiff(cast_p2);
		} else {
			if (model.equalsIgnoreCase("STE")) {
				return ste_Diff(p1, p2);
			} else {
				throw new InvalidModelException();
			}
		}
		
	}

	private double ste_Diff(Params p1, Params p2) {
		return p1.topicDiff(p2);
	}

	private double square(double d) {
		return Math.pow(d, 2);
	}
}
