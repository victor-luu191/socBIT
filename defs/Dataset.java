package defs;

import org.apache.commons.math3.linear.NonSquareMatrixException;
import org.apache.commons.math3.linear.RealMatrix;

public class Dataset {
	
	public RealMatrix ratings;		// user-item
	public RealMatrix edge_weights;	// user-user square matrix, for any two unconnected users the weight is 0
	
	// derived fields
	public int numUser;
	public int numItem;
	public int numBrand;
	
	/**
	 * Precond: {@code edge_weight} is a square matrix and {@code nrow(edge_weight) = nrow(rating)} 
	 * @param rating
	 * @param edge_weight
	 */
	public Dataset(RealMatrix rating, RealMatrix edge_weight) {
		
		super();
		
		if (! edge_weight.isSquare()) {
			System.out.println("the edge weight matrix is NOT square!!!");
			throw new NonSquareMatrixException(edge_weight.getColumnDimension(), edge_weight.getRowDimension());
		} else {
			if (edge_weight.getRowDimension() != rating.getRowDimension()) {
				System.out.println("The edge weight matrix and the rating matrix must have the same number of rows "
									+ "(the number of users)");
			} else {
				this.ratings = rating;
				this.edge_weights = edge_weight;
				numUser = rating.getRowDimension();
				numItem = rating.getColumnDimension();
			}
		}
	}
	
	
}
