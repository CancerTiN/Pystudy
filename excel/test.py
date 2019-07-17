# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import pandas as pd
xls_file = pd.ExcelFile('excel.xls')
xls_file.parse()