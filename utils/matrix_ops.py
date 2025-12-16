
class MatrixOperations:
    """
    Centralized matrix operations library.
    Provides standard matrix arithmetic without external dependencies (like numpy).
    """
    
    @staticmethod
    def multiply(A, B):
        """Matrix multiplication: C = A @ B"""
        rows_A = len(A)
        cols_A = len(A[0]) if rows_A > 0 else 0
        rows_B = len(B)
        cols_B = len(B[0]) if rows_B > 0 else 0
        
        # Check dimensions for safety (though simplified versions might skip this)
        if cols_A != rows_B:
            # For robustness, we might want to log this but continue if possible, 
            # or strictly raise ValueError. The existing code in MLP raises it.
            raise ValueError(f"Matrix dimensions don't match: ({rows_A}, {cols_A}) and ({rows_B}, {cols_B})")
        
        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        
        return result
    
    @staticmethod
    def add(A, B):
        """Matrix addition with broadcasting support"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        if len(B) == 1:  # Broadcasting: B is 1xN (bias vector)
            for i in range(rows):
                for j in range(cols):
                    result[i][j] = A[i][j] + B[0][j]
        else:  # Normal element-wise addition
            for i in range(rows):
                for j in range(cols):
                    result[i][j] = A[i][j] + B[i][j]
        
        return result
    
    @staticmethod
    def subtract(A, B):
        """Matrix subtraction: C = A - B"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] - B[i][j]
        
        return result
    
    @staticmethod
    def transpose(A):
        """Matrix transpose: A^T"""
        if not A or not A[0]:
            return [[]]
        rows = len(A)
        cols = len(A[0])
        result = [[0.0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                result[j][i] = A[i][j]
        return result
    
    @staticmethod
    def scalar_multiply(A, scalar):
        """Multiply matrix by scalar: C = scalar * A"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] * scalar
        
        return result
    
    @staticmethod
    def element_multiply(A, B):
        """Element-wise multiplication: C = A ⊙ B (Hadamard product)"""
        rows = len(A)
        cols = len(A[0]) if rows > 0 else 0
        
        result = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                result[i][j] = A[i][j] * B[i][j]
        
        return result
    
    @staticmethod
    def sum_all(A):
        """Sum all elements in matrix"""
        total = 0.0
        for row in A:
            for val in row:
                total += val
        return total

    @staticmethod
    def sum_axis0(A):
        """Sum along axis 0 (column sums) - returns 1xN"""
        if not A or not A[0]:
            return [[]]
        
        cols = len(A[0])
        result = [[0.0 for _ in range(cols)]]
        
        for i in range(len(A)):
            for j in range(cols):
                result[0][j] += A[i][j]
                
        return result

    @staticmethod
    def argmax_axis1(A):
        """Find argmax along axis 1 (row-wise max index)"""
        result = []
        for row in A:
            if row:
                max_idx = 0
                max_val = row[0]
                for j in range(1, len(row)):
                    if row[j] > max_val:
                        max_val = row[j]
                        max_idx = j
                result.append(max_idx)
            else:
                result.append(0)
        return result
