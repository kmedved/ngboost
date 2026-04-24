
install:
	pip install poetry==2.0.1 && poetry install

install-bench:
	pip install poetry==2.0.1 && poetry install --with bench

package:
	poetry build

publish: package
	source .env && poetry config pypi-token.pypi $$PYPI_TOKEN && poetry publish

lint:
	poetry run pre-commit run --hook-stage manual --all-files

test:
	poetry run pytest --slow -v

bench:
	poetry run python -m benchmarks.run --suite short --part all --no-show-plots

bench-full:
	poetry run python -m benchmarks.run --suite full --part all --no-show-plots

bench-extended:
	poetry run python -m benchmarks.run --suite full --part all --include-extended-datasets --no-show-plots

clean:
	rm -rf dist/*
