from setuptools import find_packages, setup

# This follows the style of Jaxlib installation here:
# https://github.com/google/jax#pip-installation
PYTHON_VERSION = "cp36"
CUDA_VERSION = "cuda100"  # alternatives: cuda90, cuda92, cuda100, cuda101
PLATFORM = "linux_x86_64"  # alternatives: linux_x86_64
BASE_URL = "https://storage.googleapis.com/jax-releases"


def jax_artifact(version, gpu=False):
  if gpu:
    prefix = f"{BASE_URL}/{CUDA_VERSION}/jaxlib"
    wheel_suffix = f"{PYTHON_VERSION}-none-{PLATFORM}.whl"
    location = f"{prefix}-{version}-{wheel_suffix}"
    return f"jaxlib @ {location}"

  return f"jaxlib=={version}"


def readme():
  try:
    with open('README.md') as rf:
      return rf.read()
  except FileNotFoundError:
    return None


def with_versioneer(f, default=None):
  """Attempts to execute the supplied single-arg function by passing it
versioneer if available; else, returns the default.

  """
  try:
    import versioneer
    return f(versioneer)
  except ModuleNotFoundError:
    return default


JAXLIB_VERSION = "0.1.43"
JAX_VERSION = "0.1.62"
REQUIRED_PACKAGES = [
    "blueshift-uv @ git+https://source.developers.google.com/p/blueshift-research/r/uv#egg=blueshift-uv",
    "numpy>=1.18.0",
    "tqdm>=4.42.1",
    "fs",
    "fs-gcsfs",
    f"jax=={JAX_VERSION}",
    "matplotlib",
    "sympy",
]

setup(
    name='catenary',
    version=with_versioneer(lambda v: v.get_version()),
    cmdclass=with_versioneer(lambda v: v.get_cmdclass(), {}),
    description='Hunting for new string theories.',
    long_description=readme(),
    author='Blueshift Team',
    author_email='samritchie@google.com',
    url='https://team.git.corp.google.com/blueshift/catenary',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "cpu": [jax_artifact(JAXLIB_VERSION, gpu=False)],
        "gpu": [jax_artifact(JAXLIB_VERSION, gpu=True)],
    },
    include_package_data=True,
)
