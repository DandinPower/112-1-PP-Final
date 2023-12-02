import sparse_mm

class ExtensionHandler(object):
    @staticmethod
    def sparse_mm(sparse_matrix, sparse_matrix_1):
        return sparse_mm.sparse_mm(sparse_matrix, sparse_matrix_1)