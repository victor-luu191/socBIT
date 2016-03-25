package core;

import helpers.Checkers;
import helpers.ParamUpdater;
import helpers.UtilFuncs;

import java.io.IOException;
import java.util.Optional;

import myUtil.TimeUtil;

import org.apache.commons.math3.linear.RealMatrix;

import defs.Dataset;
import defs.Hypers;
import defs.InvalidModelException;
import defs.NonConvergeException;
import defs.ParamModelMismatchException;
import defs.Params;
import defs.Model;
import defs.SoRecParams;
import defs.SocBIT_Params;

public class Trainer {
	
	private static final double EPSILON = 1;
//	private static final double INVERSE_STEP = 0.5;
	private static final double GAMMA = Math.pow(10, -4);
	private static final double EPSILON_STEP = Math.pow(2, -10);
	
	Dataset ds;
	
	// settings of this trainer
	String model;
	int numTopic;
	Hypers hypers;
	private int maxIter;
	private double stepSize;
	private RecSysCal calculator;
	
	public Trainer(String model, Dataset ds, int numTopic, Hypers hypers, int maxIter) throws InvalidModelException {
		this.model = model;
		this.ds = ds;
		this.numTopic = numTopic;
		this.hypers = hypers;
		this.maxIter = maxIter;
		calculator = buildCalculator(model);
	}
	
	/**
	 * @param initParams
	 * @param resDir
	 * @return local optimal parameters which give a local minimum of the objective function (minimum errors + regularizers)
	 * @throws IOException, InvalidModelException and  ParamModelMismatchException
	 * @throws NonConvergeException 
	 */
	Model trainByGD(Params initParams) throws IOException, InvalidModelException, ParamModelMismatchException, NonConvergeException {
		
		int numIter = 0;
		Params cParams = buildParams(initParams, model);
		double cValue = calculator.objValue(initParams);
		
		double ratingError = getRatingError(cParams);
		
		System.out.println("iter, objValue, ratingErr");
		System.out.println(numIter + ", " + cValue + "," + ratingError);
		double difference = Double.POSITIVE_INFINITY;
		
		GradCal gradCal = buildGradCal(model);
		// while not convergence and still can try more
		while ( isLarge(difference) && (numIter < maxIter) ) {
			numIter ++;
			Params cGrad = gradCal.calculate(cParams);

			
			Params nParams = lineSearch(cParams, cGrad, cValue);
			double nValue = calculator.objValue(nParams);
			
			difference = nValue - cValue;
			
			// prep for next iter
			cParams = buildParams(nParams, model);						
			cValue = nValue;
			ratingError = getRatingError(cParams);
			System.out.println(numIter + "," + cValue + ", " + ratingError);
		}
		
		if (!isLarge(difference)) {
			printConvergeMsg();
//			if (model.equalsIgnoreCase("socBIT")) {
//				edgeWeightErr = Optional.of(getEdgeWeightErr(cParams));
//			}
			
		} 
		else {
//			throw new NonConvergeException();
			String msg = "Not converged yet but already exceeded the maximum number of iterations. "
					+ "Gradient descent stopped!";
			
			System.out.println(msg);
		}
		
		Optional<Double> edgeWeightErr =  Optional.empty();
		edgeWeightErr = Optional.of(getEdgeWeightErr(cParams));
		return new Model(cParams, calculator, ratingError, edgeWeightErr, cValue);
	}
	

	private Double getEdgeWeightErr(Params params) {
		
		RealMatrix edgeWeightErrors = null;
		if (params instanceof SocBIT_Params) {
			SocBIT_Cal castCal = (SocBIT_Cal) calculator;
			edgeWeightErrors = castCal.calEdgeWeightErrors((SocBIT_Params) params);
		}
		
		if (params instanceof SoRecParams) {
			SoRec_Cal castCal = (SoRec_Cal) calculator;
			edgeWeightErrors = castCal.calEdgeWeightErrors((SoRecParams) params);
		}
		
		return UtilFuncs.sqFrobNorm(edgeWeightErrors);
	}

	private double getRatingError(Params params) {
		RealMatrix rating_errors = calculator.calRatingErrors(params);
		double sqError = square(rating_errors.getFrobeniusNorm());
		return sqError;
	}

	private Params lineSearch(Params cParams, Params cGrad, double cValue) throws ParamModelMismatchException, InvalidModelException {
		
		System.out.println("Performing line search ...");
//		System.out.println("function diff, squared params diff, necessary reduction amount" );
		
		stepSize = 1;
		Params nParams = null;	// buildParams(cParams, this.model)
		boolean sufficentReduction = false;
			
		while (!sufficentReduction && (stepSize > EPSILON_STEP)) {
			stepSize = stepSize/2 ;
			nParams = ParamUpdater.update(cParams, stepSize, cGrad, this.model);
			// todo: may need some projection here to guarantee some constraints
			double nValue = calculator.objValue(nParams);
			double funcDiff = nValue - cValue;
			double sqParamDiff = sqDiff(nParams, cParams);
			double reduction = - GAMMA/stepSize * sqParamDiff;
			
//			System.out.println(funcDiff + ", " + sqParamDiff + "," + reduction);
			
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
	
	private GradCal buildGradCal(String model) {
		
		GradCal gradCal = null;
		if (model.equalsIgnoreCase("socBIT")) {
			gradCal = new SocBIT_GradCal(this);
		}
		if (model.equalsIgnoreCase("soRec")) {
			gradCal = new SoRec_GradCal(this);
		}
		
//		if (model.equalsIgnoreCase("STE")) {
//			gradCal = new STE_GradCal(this);
//		}
//		if (model.equalsIgnoreCase("bSTE")) {
//			gradCal = new BrandSTE_GradCal(this);
//		}
		return gradCal;
	}

	private RecSysCal buildCalculator(String model) throws InvalidModelException {
		
		if (Checkers.isValid(model)) {
			if (model.equalsIgnoreCase("socBIT")) {
				calculator = new SocBIT_Cal(ds, hypers);
			} 
			
			if (model.equalsIgnoreCase("soRec")) {
				calculator = new SoRec_Cal(ds, hypers);
			}
			
//			if (model.equalsIgnoreCase("STE")) {
//				calculator = new STE_Cal(ds, hypers);
//			}
//			
//			if (model.equalsIgnoreCase("bSTE")) {
//				calculator = new BrandSTE_Cal(ds, hypers);
//			}
			
			return calculator;
		}
		
		else {
			throw new InvalidModelException();
		}
	}

	private Params buildParams(Params params, String model) throws ParamModelMismatchException, InvalidModelException {
		
		Params castParams = null;
		if (Checkers.isValid(model)) {
			if (model.equalsIgnoreCase("soRec")) {
				castParams = (SoRecParams) params; 
			}
			
			if (model.equalsIgnoreCase("socBIT") ) {
				castParams =  (SocBIT_Params) params;
			}
			
			return castParams;
		}
		
		else {
			throw new InvalidModelException();
		}
		
//		if (params instanceof SocBIT_Params) {
//		castParams =  (SocBIT_Params) params;	
//	} 
//	else {
//		String msg = "Input params type and model mismatch!!!";
//		throw new ParamModelMismatchException(msg);
//	}
		// model.equalsIgnoreCase("bSTE"), model.equalsIgnoreCase("STE")
	}

	// wrapper for computing squared difference bw two parameters where the computation depends on specific model
	private double sqDiff(Params p1, Params p2) throws InvalidModelException {
		
		double diff = 0;
		if (model.equalsIgnoreCase("socBIT") ) {// model.equalsIgnoreCase("bSTE")
			SocBIT_Params cast_p1 = (SocBIT_Params) p1;
			SocBIT_Params cast_p2 = (SocBIT_Params) p2;
			diff = cast_p1.sqDiff(cast_p2);
		} 
		
		if (model.equalsIgnoreCase("soRec")) {
			SoRecParams soRecParams1 = (SoRecParams) p1;
			SoRecParams soRecParams2 = (SoRecParams) p2;
			diff = soRecParams1.sqDiff(soRecParams2);
		}
		
		return diff;
		
//		if (model.equalsIgnoreCase("STE")) {
//			diff = p1.topicDiff(p2);
//		} else {
//			throw new InvalidModelException();
//		}
		
		
	}

	private void printConvergeMsg() {
		System.out.println("Converged to a local minimum :)");
		System.out.println("Training done.");
		System.out.println();
	}

	private boolean isLarge(double difference) {
		return Math.abs(difference) > EPSILON;
	}

	private double square(double d) {
		return Math.pow(d, 2);
	}
}
