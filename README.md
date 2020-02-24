# Catenary

![http://dataphys.org/list/gaudis-hanging-chain-models/](http://dataphys.org/list/wp-content/uploads/2015/01/IMG_5393_640.jpg "Gaudi's hanging chain models")

Catenary is a research codebase exploring an implementation of the ideas in
Henry Lin's [Bootstraps to Strings](https://arxiv.org/abs/2002.08387) paper.

The code is organized as follows:

- The `mathematica` directory contains the original Mathematica code used to
  generate the figures in the paper.
- The `catenary` directory contains (or will contain!) our Python code.

## Developing in Catenary

So you want to add some code to `catenary`. Excellent!

### Checkout and pre-commit hooks

First, check out the repo:

```
git clone sso://team/blueshift/catenary
```

Then run this command to install a special pre-commit hook that Gerrit needs to
manage code review properly. You'll only have to run this once.

```bash
f=`git rev-parse --git-dir`/hooks/commit-msg ; mkdir -p $(dirname $f) ; curl -Lo $f https://gerrit-review.googlesource.com/tools/hooks/commit-msg ; chmod +x $f
```

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your python is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks,
install `pre-commit` if you don't yet have it. I prefer using
[pipx](https://github.com/pipxproject/pipx) so that `pre-commit` stays globally
available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, you can run
either of these commands:

```bash
pre-commit run --all-files

# or this, if you've previously run `make build`:
make lint
```

### Aliases

You might find these aliases helpful when developing in Catenary:

```
[alias]
	review = "!f() { git push origin HEAD:refs/for/${1:-master}; }; f"
	amend  = "!f() { git add . && git commit --amend --no-edit; }; f"
```

### New Feature Workflow

To add a new feature, you'll want to do the following:

- create a new branch off of `master` with `git checkout -b my_branch_name`.
  Don't push this branch yet!
- run `make build` to set up a virtual environment inside the current directory.
- periodically run `make pytest` to check that your modifications pass tests.
- to run a single test file, run the following command:

```bash
env/bin/pytest tests/path/to/your/test.py
```

You can always use `env/bin/python` to start an interpreter with the correct
dependencies for the project.

When you're ready for review,

- commit your code to the branch (multiple commits are fine)
- run `git review` in the terminal. (This is equivalent to running `git push
  origin HEAD:refs/for/master`, but way easier to remember.)

The link to your pull request will show up in the terminal.

If you need to make changes to the pull request, navigate to the review page and
click the "Download" link at the bottom right:

![](https://screenshot.googleplex.com/4BP8v3TWq4R.png)

Copy the "checkout" code, which will look something like this:

```bash
git fetch "sso://team/blueshift/catenary" refs/changes/87/670987/2 && git checkout FETCH_HEAD
```

And run that in your terminal. This will get you to a checkout with all of your
code. Make your changes, then run `git amend && git review` to modify the pull
request and push it back up. (Remember, these are aliases we declared above.)

## Running the Tests

To run the tests, first run:

```bash
make build
```

To get your local environment set up. Then, run:

```bash
make pytest
```

to run our suite of tests.

## Trouble?

Get in touch with [samritchie@x.team](mailto:samritchie@x.team).
