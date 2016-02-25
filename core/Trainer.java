package core;

import helpers.ParamUpdater;
import helpers.UtilFuncs;

import java.io.IOException;

import org.apache.commons.math3.linear.RealMatrix;

import myUtil.Savers;
import defs.Dataset;
import defs.Hypers;
import defs.InvalidModelException;
import defs.ParamModelMismatchException;

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
	 */
	Params gradDescent(Params initParams, String resDir) throws IOException, InvalidModelException, ParamModelMismatchException {
		
		printStartMsg();
		int numIter = 0;
//		StringBuilder sbParams = new StringBuilder("iter, ...");
		StringBuilder sbObjValue = new StringBuilder("iter, obj_value (rating + edge_weight errors + regs), rating errors \n");
		
		Params cParams = buildParams(initParams, model);
		double cValue = calculator.objValue(initParams);
		sbObjValue = sbObjValue.append(numIter + "," + cValue + "\n");
		
		double sqError = getRatingError(cParams);
		
		System.out.println(numIter + ", " + cValue + "," + sqError);
		double difference = Double.POSITIVE_INFINITY;
		
		GradCal gradCal = buildGradCal(model);
		// while not convergence and still can try more
		while ( isLarge(difference) && (numIter < maxIter) ) {
			numIter ++;
			Params cGrad = gradCal.calculate(cParams);
			Params nParams = lineSearch(cParams, cGrad, cValue);
			double nValue = calculator.objValue(nParams);
			sbObjValue = sbObjValue.append(numIter + "," + nValue + "\n");
			difference = nValue - cValue;
			
			// prep for next iter
			cParams = buildParams(nParams, model);						
			cValue = nValue;
			sqError = getRatingError(cParams);
			System.out.println(numIter + "," + cValue + ", " + sqError);
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

	private double getRatingError(Params params) {
		RealMatrix rating_errors = calculator.calRatingErrors(params);
		double sqError = UtilFuncs.square(rating_errors.getFrobeniusNorm());
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
		// TODO Auto-generated method stub
		GradCal gradCal = null;
		if (model.equalsIgnoreCase("socBIT")) {
			gradCal = new SocBIT_GradCal(this);
		}
		if (model.equalsIgnoreCase("STE")) {
			gradCal = new STE_GradCal(this);
		}
		
		return gradCal;
	}

	private RecSysCal buildCalculator(String model) throws InvalidModelException {
		
		if (model.equalsIgnoreCase("socBIT")) {
			return new SocBIT_Cal(ds, hypers);
		} else {
			if (model.equalsIgnoreCase("STE")) {
				return new STE_Cal(ds, hypers);
			} else {
				throw new InvalidModelException();
			}
		}
	}

	private Params buildParams(Params params, String model) throws ParamModelMismatchException, InvalidModelException {
		
		if (model.equalsIgnoreCase("STE")) {
			return params;
		} 
		else {
			if (model.equalsIgnoreCase("socBIT")) {
				if (params instanceof SocBIT_Params) {
					return  (SocBIT_Params) params;	// new SocBIT_Params((SocBIT_Params) params)
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

	// wrapper for computing squared difference bw two parameters where the computation depends on specific model
	private double sqDiff(Params p1, Params p2) throws InvalidModelException {

		if (model.equalsIgnoreCase("socBIT")) {
			SocBIT_Params cast_p1 = (SocBIT_Params) p1;
			SocBIT_Params cast_p2 = (SocBIT_Params) p2;
			return cast_p1.sqDiff(cast_p2);
		} else {
			if (model.equalsIgnoreCase("STE")) {
				return p1.topicDiff(p2);
			} else {
				throw new InvalidModelException();
			}
		}
		
	}

	private void printStartMsg() {
		System.out.println("iter, obj_value (rating + edge_weight errors + regs), rating errors");
	}

	private void printConvergeMsg() {
		System.out.println("Converged to a local minimum :)");
		System.out.println("Training done.");
		System.out.println();
	}

	private boolean isLarge(double difference) {
		return Math.abs(difference) > EPSILON;
	}
}
