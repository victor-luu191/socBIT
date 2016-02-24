package core;

import helpers.UtilFuncs;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Params {

	public RealMatrix topicUser;
	public RealMatrix topicItem;

	public Params(int numUser, int numItem, int numTopic) {

		topicUser = new Array2DRowRealMatrix(numTopic, numUser);
		topicItem = new Array2DRowRealMatrix(numTopic, numItem);
	}

	Params(RealMatrix topicUser, RealMatrix topicItem) {

		this.topicItem = topicItem.copy();
		this.topicUser = topicUser.copy();
	}

	public Params(Params params) {
		topicItem = params.topicItem.copy();
		topicUser = params.topicUser.copy();
	}

	protected void initItemTopicFeats(int numItem, int numTopic) {

		// RealVector unitVector = unitVector(numTopic);
		// RealVector smallVector = unitVector.mapMultiply(EPSILON);

		RealVector uniformVector = uniformVector(numTopic);
		for (int i = 0; i < numItem; i++) {
			topicItem.setColumnVector(i, uniformVector); // unitVector
		}
	}

	protected void initUserTopicFeats(int numUser, int numTopic) {

		// RealVector unitVector = unitVector(numTopic);
		// RealVector smallVector = unitVector.mapMultiply(EPSILON);
		RealVector uniformVector = uniformVector(numTopic);

		for (int u = 0; u < numUser; u++) {
			topicUser.setColumnVector(u, uniformVector); // unitVector
		}
	}

	protected void createFeatsUniformly() {
		int numTopic = topicItem.getRowDimension();
		RealVector uniformVector = uniformVector(numTopic);

		int numItem = topicItem.getColumnDimension();
		for (int i = 0; i < numItem; i++) {
			topicItem.setColumnVector(i, uniformVector);
		}

		int numUser = topicUser.getColumnDimension();
		for (int u = 0; u < numUser; u++) {
			topicUser.setColumnVector(u, uniformVector); // unitVector
		}
	}

	private RealVector uniformVector(int size) {
		RealVector unifVector = new ArrayRealVector(size);
		unifVector = unifVector.mapAdd(1.0 / size);
		return unifVector;
	}

	protected RealVector unitVector(int size) {
		RealVector unitVector = new ArrayRealVector(size);
		int last = size - 1;
		unitVector.setEntry(last, 1);
		for (int k = 0; k < last; k++) {
			unitVector.setEntry(k, 0);
		}
		return unitVector;
	}

	double topicDiff(Params other) {
		
		double topicDiff = 0;
		RealMatrix userTopicDiff = this.topicUser.subtract(other.topicUser);
		topicDiff += sqFrobNorm(userTopicDiff);
		RealMatrix itemTopicDiff = this.topicItem.subtract(other.topicItem);
		topicDiff += sqFrobNorm(itemTopicDiff);
		return topicDiff;
	}

	private double sqFrobNorm(RealMatrix mat) {
		return UtilFuncs.square(mat.getFrobeniusNorm());
	}
	
	
}
