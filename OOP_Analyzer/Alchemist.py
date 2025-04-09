import uuid
import pandas as pd
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from sqlalchemy import ForeignKey, String, create_engine, text, select
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional
# declaring a shorthand for the declarative base class
class Base(DeclarativeBase):
    pass

# defining the classes for our project with the correct meta data
class DataSet(Base):
    __tablename__ = "dataset"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str]
    description: Mapped[Optional[str]]


    eegs: Mapped[List["EEG"]] = relationship(back_populates="dataset")

class EEG(Base):
    __tablename__ = "eeg"

    id: Mapped[str] = mapped_column(primary_key=True)
    dataset_id = mapped_column(ForeignKey("dataset.id"))
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filetype: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]]

    dataset: Mapped[DataSet] = relationship(back_populates='eegs')
    metric_sets: Mapped[List['MetricSet']] = relationship(back_populates='eegs')

class MetricSet(Base):
    __tablename__ = "metricset"

    id: Mapped[str] = mapped_column(primary_key=True)
    eeg_id = mapped_column(ForeignKey("eeg.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]]
    signal_len: Mapped[Optional[int]]
    fs: Mapped[Optional[int]]
    window_len: Mapped[Optional[int]]
    window_overlap: Mapped[Optional[int]]
    lower_cutoff: Mapped[Optional[float]]
    upper_cutoff: Mapped[Optional[float]]
    montage: Mapped[Optional[str]]

    eegs: Mapped[List['EEG']] = relationship(back_populates='metric_sets')
    metrics: Mapped[List['Metric']] = relationship(back_populates='metric_set')

class Metric(Base):
    __tablename__ = "metric"

    id: Mapped[str] = mapped_column(primary_key=True)
    metric_set_id = mapped_column(ForeignKey("metricset.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    result_path: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]]

    metric_set: Mapped[MetricSet] = relationship(back_populates='metrics')

# functions to deal with the data objects
def add_result_data_table(engine, tablename: str):
    class MetricData(Base):
        __tablename__ = tablename

        id: Mapped[str] = mapped_column(primary_key=True)
        metric_id = mapped_column(ForeignKey("metric.id"), nullable=False)
    Base.metadata.create_all(bind=engine)

def remove_table(engine, table_name: str, del_from_metadata = True):
    try:
        # Execute the DROP TABLE command
        stmt = text(f'DROP TABLE IF EXISTS {table_name}')
        with engine.connect() as connection:
            connection.execute(stmt)
            print(f"Table {table_name} removed successfully.")

        # Remove the table from the MetaData object
        table = Base.metadata.tables.get(table_name)
        if table is not None and del_from_metadata:
            Base.metadata.remove(table)
            print(f"Table {table_name} removed from metadata successfully.")
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
    :param column_names: List of tuples, where each tuple contains the column name and column type.
    :param column_type: sql type to apply to the added columns (should be the type of the data inserted)
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

def adding_data(engine, table_name:str, parameter_id: str, data: pd.DataFrame):
    add_result_data_table(engine, table_name)
    try:
        # Add string UUID and parameter_id to the DataFrame
        data['id'] = [str(uuid.uuid4()) for _ in range(len(data))]
        # data['id'] = 1
        data['parameter_id'] = parameter_id
        # Add a new table if it doesn't exist and insert data
        data.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Data inserted into table {table_name} successfully.")
    except SQLAlchemyError as e:
        print(f"Error: {e}")

def get_column_value_pairs(orm_object):
    """
    Retrieve column-value pairs from an SQLAlchemy ORM object as a dictionary.

    :param orm_object: The SQLAlchemy ORM object.
    :return: A dictionary containing column-value pairs.
    """
    table_class = type(orm_object)
    column_value_pairs = {column.name: getattr(orm_object, column.name) for column in table_class.__table__.columns}
    return column_value_pairs

def check_and_add_entry(engine, table_class, entry_data: dict, add_if_exists: bool = False):
    """
    Check if an entry exists in the table and add it based on the add_if_exists flag.

    :param engine: SQLAlchemy engine connected to the database.
    :param table_class: The ORM class representing the table.
    :param entry_data: Dictionary containing the entry data (excluding the ID).
    :param add_if_exists: Boolean flag indicating whether to add the entry if it already exists.
    """
    # Remove the 'id' from entry_data for comparison
    entry_data_no_id = {k: v for k, v in entry_data.items() if k != 'id'}

    # Check if the entry exists
    existing_entries = find_entries(engine, table_class, **entry_data_no_id)

    if existing_entries and not add_if_exists:
        print("Entry already exists and will not be added.")
        return None

    # Add the entry if it doesn't exist or if add_if_exists is True
    new_entry = table_class(**entry_data)
    try:
        with Session(engine) as session:
            session.add(new_entry)
            session.commit()
            print("Entry added successfully.")
    except SQLAlchemyError as e:
        print(f"Error: {e}")

    return new_entry

def test_dbcreation(db_path:str =None):
    engine = initialize_tables(db_path)
    add_result_data_table(engine, 'test')
    add_column(engine, 'test', 'test', 'INTEGER')
    remove_column(engine, 'test', 'test')
    add_multiple_columns(engine, 'test', ['test1', 'test2', 'test3'], 'INTEGER')
    # TODO: doesnt seem to work correctly yet
    # Here the table does get removed but for some reason it seems to be readded later, have to investigate
    remove_table(engine, 'test')
    return engine

def test_adding_data(engine):
    # create mock data
    data = pd.DataFrame(columns=['a', 'b', 'c'], data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = DataSet(
        id=str(uuid.uuid4()),
        name='EEG Study Dataset',
        path='/data/eeg_study',
        description='Dataset for EEG study on cognitive functions'
    )
    eeg = EEG(
        id=str(uuid.uuid4()),
        filename='subject_01_session_01',
        filetype='.edf',
        filepath='/data/eeg_study',
        description='EEG recording for subject 01, session 01'
    )
    metric_set = MetricSet(
        id=str(uuid.uuid4()),
        name='Alpha Band Power Analysis',
        description='Analysis of alpha band power for cognitive task performance'
    )
    metric_id = str(uuid.uuid4())
    metric = Metric(
        id=metric_id,
        name='Alpha Power',
        result_path='/results/alpha_power_subject_01_session_01.csv',
        description='Computed alpha power for subject 01, session 01'
    )
    dataset.eegs.append(eeg)
    eeg.metric_sets.append(metric_set)
    metric_set.metrics.append(metric)
    with Session(engine) as session:
        session.add(dataset)
        session.add(eeg)
        session.add(metric_set)
        session.add(metric)
        session.commit()
    adding_data(engine, 'test_data', metric_id, data)


if __name__ == "__main__":
    db_path = 'test.sqlite'
    engine = test_dbcreation(db_path)
    test_adding_data(engine)