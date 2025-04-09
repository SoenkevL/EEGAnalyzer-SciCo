import uuid
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from sqlalchemy import ForeignKey, String, create_engine, text, select, DateTime, func
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional
# declaring a shorthand for the declarative base class
class Base(DeclarativeBase):
    pass


# defining the classes for our project with the correct meta data
class DataSet(Base):
    __tablename__ = "dataset"

    id: Mapped[str] = mapped_column(primary_key=True)
    last_altered: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str]
    description: Mapped[Optional[str]]

    eegs: Mapped[List["EEG"]] = relationship(back_populates="dataset")

class EEG(Base):
    __tablename__ = "eeg"

    id: Mapped[str] = mapped_column(primary_key=True)
    dataset_id = mapped_column(ForeignKey("dataset.id"))
    last_altered: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
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
    last_altered: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    name: Mapped[str] = mapped_column(String, nullable=False)
    result_path: Mapped[str]
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
    last_altered: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    name: Mapped[str] = mapped_column(String, nullable=False)  # Name of the metric (e.g., 'entropy_permutation')
    description: Mapped[Optional[str]]

    metric_set: Mapped[MetricSet] = relationship(back_populates='metrics')
    # Relationship to metric data will be handled dynamically

# functions to deal with the data objects
def add_result_data_table(engine, metric_id: str, channel_names: list):
    """
    Create a table to store metric data for a specific metric.

    Parameters:
    - engine: SQLAlchemy engine
    - metric_id: ID of the metric this data belongs to
    - channel_names: List of channel names to create columns for
    """
    # Create a unique table name based on the metric ID
    tablename = f"metric_data_{metric_id.replace('-', '_')}"

    # Define the basic MetricData class with essential columns including window/epoch info
    class MetricData(Base):
        __tablename__ = tablename

        id: Mapped[str] = mapped_column(primary_key=True)
        metric_id = mapped_column(ForeignKey("metric.id"), nullable=False)
        start_time: Mapped[float] = mapped_column(nullable=False)  # Start time of the window in seconds
        duration: Mapped[float] = mapped_column(nullable=False)  # Duration of the window in seconds
        label: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # Label for the window (e.g., annotation name)
        last_altered: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Create the table with the basic columns
    Base.metadata.create_all(bind=engine)

    # Prepare safe column names for channels
    safe_channel_names = []
    for channel_name in channel_names:
        # Create a safe column name (replace special characters)
        safe_name = channel_name.replace('-', '_').replace(' ', '_')
        safe_channel_names.append(safe_name)

    # Add channel columns using the existing function
    if safe_channel_names:
        add_multiple_columns(engine, tablename, safe_channel_names, 'FLOAT')

    return tablename

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

def adding_metric_data(engine, metric_id: str, channel_data: dict, start_time: float, duration: float, label: str = None):
    """
    Add metric data for a specific metric and channels.

    Parameters:
    - engine: SQLAlchemy engine
    - metric_id: ID of the metric this data belongs to
    - channel_data: Dictionary mapping channel names to their metric values
    - start_time: Start time of the window in seconds
    - duration: Duration of the window in seconds
    - label: Label for the window (e.g., annotation name)
    """
    try:
        # Create a unique table name based on the metric ID
        tablename = f"metric_data_{metric_id.replace('-', '_')}"

        # Check if the table exists, if not create it
        with engine.connect() as conn:
            table_exists = conn.execute(text(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tablename}'"
            )).scalar() is not None

            if not table_exists:
                # Create the table with the appropriate columns
                channel_names = list(channel_data.keys())
                add_result_data_table(engine, metric_id, channel_names)
            else:
                # Check if all required columns exist, add any missing ones
                existing_columns = [row[1] for row in conn.execute(text(f"PRAGMA table_info({tablename})")).fetchall()]
                missing_columns = []

                for channel in channel_data.keys():
                    safe_name = channel.replace('-', '_').replace(' ', '_')
                    if safe_name not in existing_columns:
                        missing_columns.append(safe_name)

                # Add any missing columns
                if missing_columns:
                    add_multiple_columns(engine, tablename, missing_columns, 'FLOAT')

        # Prepare data for insertion
        data = {
            'id': str(uuid.uuid4().hex),
            'metric_id': metric_id,
            'start_time': start_time,
            'duration': duration,
            'label': label,
            'last_altered': datetime.now()
        }

        # Add channel data
        for channel, value in channel_data.items():
            safe_name = channel.replace('-', '_').replace(' ', '_')
            data[safe_name] = value

        # Convert to DataFrame and insert
        df = pd.DataFrame([data])
        df.to_sql(tablename, con=engine, if_exists='append', index=False)

        print(f"Data inserted into table {tablename} successfully.")
        return True
    except SQLAlchemyError as e:
        print(f"Error adding metric data: {e}")
        return False

def get_column_value_pairs(orm_object):
    """
    Retrieve column-value pairs from an SQLAlchemy ORM object as a dictionary.

    :param orm_object: The SQLAlchemy ORM object.
    :return: A dictionary containing column-value pairs.
    """
    table_class = type(orm_object)
    column_value_pairs = {column.name: getattr(orm_object, column.name) for column in table_class.__table__.columns}
    return column_value_pairs


def test_dbcreation(db_path:str =None):
    engine = initialize_tables(db_path)

    # Create a test metric ID
    test_metric_id = str(uuid.uuid4().hex)

    # Test creating a metric data table with channel names
    test_channel_names = ['Fp1', 'Fp2', 'F3', 'F4']
    tablename = add_result_data_table(engine, test_metric_id, test_channel_names)
    print(f"Created test table: {tablename}")

    # Test adding an additional column
    add_column(engine, tablename, 'test_column', 'INTEGER')

    # Test removing a column
    remove_column(engine, tablename, 'test_column')

    # Test adding more columns
    add_multiple_columns(engine, tablename, ['additional1', 'additional2'], 'INTEGER')

    # List all columns in the table to verify
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"PRAGMA table_info({tablename})"))
            columns = result.fetchall()
            print(f"Columns in {tablename}:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
    except SQLAlchemyError as e:
        print(f"Error listing columns: {e}")

    # Test removing the table
    # TODO: doesnt seem to work correctly yet
    # Here the table does get removed but for some reason it seems to be readded later, have to investigate
    remove_table(engine, tablename)

    return engine

def test_adding_data(engine):
    # Create test dataset, EEG, and metric set
    dataset = DataSet(
        id=str(uuid.uuid4().hex),
        name='EEG Study Dataset',
        path='/data/eeg_study',
        description='Dataset for EEG study on cognitive functions'
        # last_altered will be set automatically
    )

    eeg = EEG(
        id=str(uuid.uuid4().hex),
        filename='subject_01_session_01',
        filetype='.edf',
        filepath='/data/eeg_study',
        description='EEG recording for subject 01, session 01'
        # last_altered will be set automatically
    )

    metric_set_id = str(uuid.uuid4().hex)
    metric_set = MetricSet(
        id=metric_set_id,
        eeg_id=eeg.id,  # Set the EEG ID explicitly
        name='Alpha Band Power Analysis',
        result_path='/results/alpha_power_subject_01_session_01.csv',
        description='Analysis of alpha band power for cognitive task performance'
        # last_altered will be set automatically
    )

    # Create a test metric
    metric_id = str(uuid.uuid4().hex)
    metric = Metric(
        id=metric_id,
        metric_set_id=metric_set.id,  # Set the metric set ID explicitly
        name='Alpha Power',
        description='Computed alpha power for subject 01, session 01'
        # last_altered will be set automatically
    )

    # Set up relationships
    dataset.eegs.append(eeg)
    eeg.metric_sets.append(metric_set)
    metric_set.metrics.append(metric)

    # Save to database
    with Session(engine) as session:
        session.add(dataset)
        session.add(eeg)
        session.add(metric_set)
        session.add(metric)
        session.commit()

    # Test adding metric data
    channel_data = {
        'Fp1': 0.75,
        'Fp2': 0.82,
        'F3': 0.65,
        'F4': 0.71
    }

    # Add the channel data to the database with window/epoch information
    adding_metric_data(
        engine=engine,
        metric_id=metric_id,
        channel_data=channel_data,
        start_time=0.0,  # Start time of the window in seconds
        duration=5.0,    # Duration of the window in seconds
        label='resting'  # Label for the window
    )

    print("Test data added successfully.")


if __name__ == "__main__":
    # Run the test functions
    print("Running database tests...")
    db_path = 'test.sqlite'
    engine = test_dbcreation(db_path)
    test_adding_data(engine)
    print("All tests completed successfully.")