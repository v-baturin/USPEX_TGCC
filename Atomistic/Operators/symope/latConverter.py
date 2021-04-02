from math import sqrt, acos, cos, sin, tan

import numpy as np


def latConverter(input_list):
    """
    The function converts lattice from 1x6 representation to 3x3 and back.
    :param input_list: input values in a list form.
    :return output_list: converted list of values (NumPy array).
    """

    output_list = None

    # input_list = np.asarray(input_list)

    def sum_squares(list2sum):
        # Sum squares of all values in the list.
        sum_value = 0.0
        for i in range(len(list2sum)):
            sum_value += float(list2sum[i]) ** 2
        return sum_value

    def sum_mult(list1, list2):
        # Sum element-wise multiplication of all values in 2 lists.
        sum_value = 0.0
        for i in range(len(list1)):
            sum_value += float(list1[i]) * float(list2[i])
        return sum_value

    if len(input_list) == 3 and len(input_list[0]) == 3:
        # 3x3 representation -> 1x6 (a, b, c, alpha, beta, gamma)
        output_list = [0.0] * 6

        output_list[0] = sqrt(sum_squares(input_list[0]))
        output_list[1] = sqrt(sum_squares(input_list[1]))
        output_list[2] = sqrt(sum_squares(input_list[2]))

        output_list[5] = acos(sum_mult(input_list[0], input_list[1]) / (output_list[0] * output_list[1]))
        output_list[4] = acos(sum_mult(input_list[0], input_list[2]) / (output_list[0] * output_list[2]))
        output_list[3] = acos(sum_mult(input_list[1], input_list[2]) / (output_list[1] * output_list[2]))

        # Complicate the output:
        output_list = [output_list]

    elif len(input_list) == 1 and len(input_list[0]) == 6:
        # 1x6 (a, b, c, alpha, beta, gamma) -> 3x3 representation

        # We need try/except here to avoid possible errors during sqrt calculation:
        try:
            # Simplify the input:
            input_list = input_list[0]

            output_list = []
            for i in range(3):
                output_list.append([0.0] * 3)

            output_list[0][0] = input_list[0]

            output_list[1][0] = input_list[1] * cos(input_list[5])
            output_list[1][1] = input_list[1] * sin(input_list[5])
            output_list[2][0] = input_list[2] * cos(input_list[4])

            output_list[2][1] = input_list[2] * cos(input_list[3]) * sin(input_list[5]) - \
                                ((input_list[2] * cos(input_list[4]) - \
                                  input_list[2] * cos(input_list[3]) * cos(input_list[5])) / tan(input_list[5]))

            output_list[2][2] = sqrt(input_list[2] ** 2 - output_list[2][0] ** 2 - output_list[2][1] ** 2)
        except ValueError:
            output_list = None

    # ---------------------------------------------------------------------------

    output_list = np.asarray(output_list)

    return output_list

"""
# -------------------------------------------------------------------------------
# Testing:
if __name__ == "__main__":
    # input_list  = [[4.5, 3.8, 10.6, 90.0, 90.0, 120.0]]
    input_list = [[4.5, 3.8, 10.6, 1.5707963267948966, 1.5707963267948966, 2.0943951023931953]]
    # input_list = [[10.1, 8.4, 12.5, 90.0, 101.3, 90.0]]
    output_list = latConverter(input_list)
    print('Input :\n', input_list)
    print('Output:\n')
    for i in output_list:
        print('%12.8f' * 3 % (i[0], i[1], i[2]))

    '''
    ================================================================================
    TEST I:

    From Matlab:

    To get started, type one of these: helpwin, helpdesk, or demo.
    For product information, visit www.mathworks.com.

    >> main

    input =

        4.5000    3.8000   10.6000   90.0000   90.0000  120.0000


    output =

        4.5000         0         0
        3.0939    2.2063         0
       -4.7496   -1.5201    9.3537

    >> 

    ************

    From Python:

    Input :
    [[4.5, 3.8, 10.6, 90.0, 90.0, 120.0]]
    Output:

      4.50000000  0.00000000  0.00000000
      3.09388769  2.20632250  0.00000000
     -4.74958033 -1.52005754  9.35365767

    ================================================================================
    TEST II:

    From Matlab:

    To get started, type one of these: helpwin, helpdesk, or demo.
    For product information, visit www.mathworks.com.

    >> main

    input =

       10.1000    8.4000   12.5000   90.0000  101.3000   90.0000


    output =

       10.1000         0         0
       -3.7638    7.5096         0
        8.9823   -1.7631    8.5124

    >> 

    ************

    From Python:

    Input :
    [[10.1, 8.4, 12.5, 90.0, 101.3, 90.0]]
    Output:

     10.10000000  0.00000000  0.00000000
     -3.76381838  7.50957197  0.00000000
      8.98227558 -1.76309327  8.51235734

    '''
    pass

    input_list = [

        [4.50000000, 0.00000000, 0.00000000],
        [-1.90000000, 3.29089653, 0.00000000],
        [0.00000000, 0.00000000, 10.60000000],
    ]

    output_list = latConverter(input_list)
    print('\n\n\nInput :\n')
    for i in input_list:
        print('%12.8f' * 3 % (i[0], i[1], i[2]))

    print('Output:', output_list)
    # for i in output_list:
    #    print '%12.6f'*6 % (i[0], i[1], i[2], i[3], i[4], i[5])

    '''
    ================================================================================
    TEST III:

    From Matlab:

    To get started, type one of these: helpwin, helpdesk, or demo.
    For product information, visit www.mathworks.com.

    >> main

    input =

        4.5000         0         0
        3.0939    2.2063         0
       -4.7496   -1.5201    9.3537


    output =

        4.5000
        3.8000
       10.6000
        2.0354
        2.0354
        0.6195

    >> 
    
    ************

    From Python:

    Input :

      4.50000000  0.00000000  0.00000000
      3.09388769  2.20632250  0.00000000
     -4.74958033 -1.52005754  9.35365767
    Output:
        4.500000    3.800000   10.600000    2.035406    2.035406    0.619479    

    '''
    pass
"""
