#!/bin/bash

set -e -x

cd /github/workspace

PYTHON_VERSION=$1
PARALLEL=$2

if [ "${PYTHON_VERSION}" = "3.6" ]; then
    PY_VER=cp36-cp36m
elif [ "${PYTHON_VERSION}" = "3.7" ]; then
    PY_VER=cp37-cp37m
elif [ "${PYTHON_VERSION}" = "3.8" ]; then
    PY_VER=cp38-cp38
elif [ "${PYTHON_VERSION}" = "3.9" ]; then
    PY_VER=cp39-cp39
elif [ "${PYTHON_VERSION}" = "3.10" ]; then
    PY_VER=cp310-cp310
elif [ "${PYTHON_VERSION}" = "3.11" ]; then
    PY_VER=cp311-cp311
elif [ "${PYTHON_VERSION}" = "3.12" ]; then
    PY_VER=cp312-cp312
elif [ "${PYTHON_VERSION}" = "3.13" ]; then
    PY_VER=cp313-cp313
elif [ "${PYTHON_VERSION}" = "3.13t" ]; then
    PY_VER=cp313-cp313t
elif [ "${PYTHON_VERSION}" = "3.14-dev" ]; then
    PY_VER=cp314-cp314
elif [ "${PYTHON_VERSION}" = "3.14t-dev" ]; then
    PY_VER=cp314-cp314t
fi

PY_EXE=/opt/python/"${PY_VER}"/bin/python3
sed -i "/DPython3_EXECUTABLE/a \                '-DPython3_EXECUTABLE=${PY_EXE}'," setup.py
sed -i "/DPython3_EXECUTABLE/a \                '-DFORCE_LIB_ABS_PATH=OFF'," setup.py

ls -l /opt/python
/opt/python/"${PY_VER}"/bin/pip install --upgrade --no-cache-dir pip setuptools
/opt/python/"${PY_VER}"/bin/pip install --no-cache-dir mkl==2024.1.0 mkl-include intel-openmp numpy 'cmake>=3.19' pybind11==2.12.0
$(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -m pip install auditwheel==5.1.2
$(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -m pip install setuptools

if [ "${PARALLEL}" = "mpi" ]; then
    yum install -y wget openssh-clients openssh-server
    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.5.tar.gz
    tar zxf openmpi-5.0.5.tar.gz
    cd openmpi-5.0.5
    ./configure --prefix=/usr/local |& tee config.out
    make -j 4 |& tee make.out
    make install |& tee install.out
    cd ..
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    /opt/python/"${PY_VER}"/bin/pip install --no-cache-dir mpi4py
    sed -i "/DUSE_MKL/a \                '-DMPI=ON'," setup.py
    sed -i "/intel-openmp/d" pyproject.toml
    sed -i "/mkl-include/d" pyproject.toml
    sed -i "/cmake>/d" pyproject.toml
    sed -i "s/name=\"block2\"/name=\"block2-mpi\"/g" setup.py
    sed -i "s/name = \"block2\"/name = \"block2-mpi\"/g" pyproject.toml
    sed -i '/for soname, src_path/a \                if any(x in soname for x in ["libmpi", "libopen-pal", "libopen-rte"]): continue' \
        $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")
    sed -i '/for soname, src_path/a \                if "libmpi.so" in soname: patcher.replace_needed(fn, soname, "libmpi.so")' \
        $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")
    sed -i '/for n in needed/a \                if "libmpi.so" in n: patcher.replace_needed(path, n, "libmpi.so")' \
        $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")
fi

sed -i '/new_soname = src_name/a \    if any(x in src_name for x in ["libmkl_avx2", "libmkl_avx512"]): new_soname = src_name' \
    $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")
${PY_EXE} -c 'import site; x = site.getsitepackages(); x += [xx.replace("site-packages", "dist-packages") for xx in x]; print("*".join(x))' > /tmp/ptmp
sed -i '/rpath_set\[rpath\]/a \    import site\n    for x in set(["../lib" + p.split("lib")[-1] for p in open("/tmp/ptmp").read().strip().split("*")]): rpath_set[rpath.replace("../..", x)] = ""' \
    $($(cat $(which auditwheel) | head -1 | awk -F'!' '{print $2}') -c "from auditwheel import repair;print(repair.__file__)")

cmake --version
/opt/python/"${PY_VER}"/bin/pip wheel . -w ./dist --no-deps

find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') --plat '${PLAT}'" \;
find . -type f -iname "*-linux*.whl" -exec rm {} \;
find . -type f -iname "*-manylinux*.whl"

rm /tmp/ptmp
