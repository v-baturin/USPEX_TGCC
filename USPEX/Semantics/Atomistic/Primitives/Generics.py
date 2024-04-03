from typing import Annotated, Literal
from numpy.typing import NDArray


Vector3 = Annotated[NDArray[float], Literal[3]]
VectorN3 = Annotated[NDArray[float], Literal['N', 3]]
Operation3 = Annotated[NDArray[float], Literal[3, 3]]
Operation4 = Annotated[NDArray[float], Literal[4, 4]]
SquareMatrixN = Annotated[NDArray[float], Literal['N', 'N']]
SquareMatrixN3 = Annotated[NDArray[float], Literal['N', 'N', 3]]
Tiling3 = Annotated[NDArray[int], Literal[3, 3]]