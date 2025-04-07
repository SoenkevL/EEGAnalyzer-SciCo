import uuid
from functools import partial
import numpy as np
import pandas as pd
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from sqlalchemy import MetaData, ForeignKey, Table, Column, Integer, String, create_engine, text, BIGINT, select
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional
# declaring a shorthand for the declarative base class
class Base(DeclarativeBase):
    pass

# defining the classes for our project with the correct meta data
# we will first go for a simple approach leaving the actual calculations in csv format as is the case now
class DataSet(Base):
    __tablename__ = "dataset"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str]
    description: Mapped[Optional[str]]


    eegs: Mapped[List["EEG"]] = relationship(back_populates="dataset")


class EEG(Base):
    __tablename__ = "EEG"

    id: Mapped[str] = mapped_column(primary_key=True)
    dataset_id = mapped_column(ForeignKey("dataset.id"))
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filetype: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)#
    description: Mapped[Optional[str]]

    dataset: Mapped[DataSet] = relationship(back_populates='eegs')
    metrics: Mapped[List['Metric']] = relationship(back_populates='eeg')


class Metric(Base):
    __tablename__ = "metric"

    id: Mapped[str] = mapped_column(primary_key=True)
    eeg_id = mapped_column(ForeignKey("EEG.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)

    eeg: Mapped[EEG] = relationship(back_populates='metrics')
    processing_params: Mapped[List['MetricParameters']] = relationship(back_populates='metric')


class MetricParameters(Base):
    __tablename__ = "metric_parameter"

    id: Mapped[str] = mapped_column(primary_key=True)
    metric_id = mapped_column(ForeignKey("metric.id"), nullable=False)
    signal_len: Mapped[Optional[int]]
    description: Mapped[Optional[str]]
    fs: Mapped[Optional[int]]
    window_len: Mapped[Optional[int]]
    window_overlap: Mapped[Optional[int]]
    lower_cutoff: Mapped[Optional[float]]
    upper_cutoff: Mapped[Optional[float]]
    montage: Mapped[Optional[str]]

    metric: Mapped[Metric] = relationship(back_populates='processing_params')
    results: Mapped[List['MetricResults']] = relationship(back_populates="metric_parameter")


class MetricResults(Base):
    __tablename__ = "metric_result"

    id: Mapped[str] = mapped_column(primary_key=True)
    parameter_id = mapped_column(ForeignKey("metric_parameter.id"), nullable=False)
    result_path: Mapped[str] = mapped_column(String, nullable=False)

    metric_parameter: Mapped[MetricParameters] = relationship(back_populates='results')
    # results_data: Mapped[List['MetricResultsData']] = relationship(back_populates='metric_result')


# functions to deal with the data objects
def add_result_data_table(engine, tablename: str):
    class MetricResultsData(Base):
        __tablename__ = tablename

        id: Mapped[str] = mapped_column(primary_key=True)
        result_id = mapped_column(ForeignKey("metric_result.id"), nullable=False)
        # metric_result: Mapped[MetricResults] = relationship(back_populates='results_data')
    Base.metadata.create_all(bind=engine)

def remove_table(engine, table_name: str):
    try:
        # Execute the DROP TABLE command
        stmt = text(f'DROP TABLE IF EXISTS {table_name}')
        with engine.connect() as connection:
            connection.execute(stmt)
            print(f"Table {table_name} removed successfully.")
    except SQLAlchemyError as e:
        print(f"Error: {e}")

def add_column(engine, table_name:str , column_name:str, column_type:str):
    try:
        # Compile the column type for the specific database dialect

        # Execute the ALTER TABLE command
        stmt = text(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}')
        with Session(engine) as session:
            result = session.execute(stmt)
            print(f"Column {column_name} added successfully.")
            session.commit()
    except SQLAlchemyError as e:
        print(f"Error: {e}")

def add_multiple_columns(engine, table_name: str, column_names: list[str], column_type: str=None):
    """
    Add multiple columns to an existing table.


    :param engine: SQLAlchemy engine connected to the database.
    :param table_name: Name of the table to which columns will be added.
    :param columns: List of tuples, where each tuple contains the column name and column type.
    :param type: sql type to apply to the added columns (should be the type of the data inserted)
    """
    try:
        with Session(engine) as session:
            for column_name in column_names:
                # Compile the column type for the specific database dialect
                # Execute the ALTER TABLE command
                if type:
                    stmt = text(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}')
                else:
                    stmt = text(f'ALTER TABLE {table_name} ADD COLUMN {column_name}')
                session.execute(stmt)
                print(f"Column {column_name} added successfully.")

            session.commit()
            print(f"Columns commited successfully.")
    except SQLAlchemyError as e:
        print(f"Error: {e}")
        
def remove_column(engine, table_name, column_name):
    try:
        # Execute the ALTER TABLE command to drop the column
        stmt = text(f'ALTER TABLE {table_name} DROP COLUMN {column_name}')
        with engine.connect() as connection:
            connection.execute(stmt)
            print(f"Column {column_name} removed successfully.")
    except SQLAlchemyError as e:
        print(f"Error: {e}")

def find_entries(engine, table_class, **kwargs):
    """
    Check if an entry exists in the table based on given parameters.

    :param engine: SQLAlchemy engine connected to the database.
    :param table_class: The ORM class representing the table.
    :param kwargs: Column-value pairs to filter the query.
    :return: True if the entry exists, False otherwise.
    """
    try:
        with Session(engine) as session:
            query = select(table_class).filter_by(**kwargs)
            result = session.scalars(query).all()
            return result
    except SQLAlchemyError as e:
        print(f"Error: {e}")
        return []

def initialize_tables(path=None, path_is_relative=True):
    if path:
        if path_is_relative:
            engine = create_engine(f"sqlite+pysqlite:///{path}")
        else:
            engine = create_engine(f"sqlite+pysqlite:////{path}")
    else:
        engine = create_engine("sqlite+pysqlite://:memory:")
    Base.metadata.create_all(bind=engine)
    return engine

def adding_data(engine, table_name:str, parameter_id: str, result_path: str, data: pd.DataFrame):
    add_result_data_table(engine, table_name)
    try:
        # Add string UUID and parameter_id to the DataFrame
        data['id'] = [str(uuid.uuid4()) for _ in range(len(data))]
        # data['id'] = 1
        data['parameter_id'] = parameter_id
        data['result_path'] = result_path
        # Add a new table if it doesn't exist and insert data
        data.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Data inserted into table {table_name} successfully.")
    except SQLAlchemyError as e:
        print(f"Error: {e}")

def test_dbcreation(db_path:str =None):
    engine = initialize_tables(db_path)
    add_result_data_table(engine, 'test')
    add_column(engine, 'test', 'test', 'INTEGER')
    remove_column(engine, 'test', 'test')
    add_multiple_columns(engine, 'test', ['test1', 'test2', 'test3'], 'INTEGER')
    remove_table(engine, 'test')
    return engine

def test_adding_data(engine):
    # create mock data
    data = pd.DataFrame(columns=['a', 'b', 'c'], data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # populate some examples into the metadata tables
    dataset = DataSet(id=str(uuid.uuid4()), name='moc_dataset', path='')
    eeg = EEG(id=str(uuid.uuid4()), filename='', filetype='EEG', filepath='')
    metric = Metric(id=str(uuid.uuid4()), name='test_metric')
    metric_param_id = str(uuid.uuid4())
    metric_params = MetricParameters(id=str(uuid.uuid4()))
    dataset.eegs.append(eeg)
    eeg.metrics.append(metric)
    metric.processing_params.append(metric_params)
    with Session(engine) as session:
        session.add(dataset)
        session.add(eeg)
        session.add(metric)
        session.add(metric_params)
        session.commit()
    adding_data(engine, 'test_data', metric_param_id, '', data)


if __name__ == "__main__":
    db_path = 'test.sqlite'
    engine = test_dbcreation(db_path)
    test_adding_data(engine)