### pose_classification
This Project is used to classify the pose of a given picture. The pipeline is as follows:
1. detect the pose landmarks;
2. obtains the embedding of pose by combining 3D landmarks and the pose distances
3. train the classifier on trainset, the input of classifier is embedding mentioned in step 2.
4. using classifier to classify a given pose


### how to use
'test.py' provide the entry of testing, including pose and expression. run:

`"python3 test.py --image-list data/test_images/pose.lst --detect-type 0"` for pose detection

`"python3 test.py --image-list data/test_images/expression.lst --detect-type 1"` for expression detection

### how to protect source code 

1. create a ubuntu(or other linux system) docker with python3.8 environment
2. install the python packages required for this project, see `requirements.txt`
3. Use cython to parse the python code into c/c++ code, and compile the c/c++ code into .so file. 

e.g. package the api_pose.py into .so as follows:

```
# setup.py
from distutils.core import setup
from Cython.Build import cythonize
setup(name='Pose recognition', ext_modules=cythonize("api_pose.py"))
```

4. After finishing running above code, we can get `api_pose.cpython-38-x86_64-linux-gnu.so` file.
once we get the .so file, we can call the function in `api_pose.py` through `api_pose.cpython-38-x86_64-linux-gnu.so`, and the `api_pose.py` can be deleted to protect source code. The same step for `api_expression.py` to generate `api_expression.cpython-38-x86_64-linux-gnu.so`

5. Since we get the corresponding *.so, we can detect the `api_expression.py` and `api_pose.py` to protect source code.
6. save the docker as image, and provide the docker image for providing test demo.

##### tips
None