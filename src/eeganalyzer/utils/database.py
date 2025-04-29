"""
Copyright (C) <2025>  <Soenke van Loh>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

Database utilities for EEG analysis.

This module provides functions for interacting with the database.
"""

import uuid
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session, sessionmaker
from sqlalchemy import ForeignKey, String, create_engine, text, select, DateTime, func, Integer, Table, Column
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional, Union, Dict, Any, Tuple, Type
# declaring a shorthand for the declarative base class
class Base(DeclarativeBase):
    pass

# defining the classes for our project with the correct meta data
class DataSet(Base):
    __tablename__ = "dataset"

    id: Mapped[str] = mapped_column(primary_key=True, default=str(uuid.uuid4().hex))
    last_altered: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str]
    description: Mapped[Optional[str]]

    eegs: Mapped[List["EEG"]] = relationship(back_populates="dataset")

class EEG(Base):
    __tablename__ = "eeg"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=str(uuid.uuid4().hex))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("dataset.id"))
    last_altered: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filetype: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]]

    dataset: Mapped[DataSet] = relationship(back_populates='eegs')
    # experiments: Mapped[List['Experiment']] = relationship(back_populates='eegs', secondary="association_table")
    experiments: Mapped[List['Experiment']] = relationship(back_populates='eegs', secondary="result_association")

class Experiment(Base):
    __tablename__ = "experiment"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=str(uuid.uuid4().hex))
    last_altered: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    metric_set_name: Mapped[str] = mapped_column(String, nullable=False)  # Name of the metric set (e.g., 'entropy')
    run_name: Mapped[str] = mapped_column(String, nullable=False)  # Name of the metric set (e.g., 'entropy')
    description: Mapped[Optional[str]]

    fs: Mapped[Optional[int]]
    start: Mapped[Optional[int]]
    stop: Mapped[Optional[int]]
    window_len: Mapped[Optional[int]]
    window_overlap: Mapped[Optional[int]]
    lower_cutoff: Mapped[Optional[float]]
    upper_cutoff: Mapped[Optional[float]]
    montage: Mapped[Optional[str]]

    # eegs: Mapped[List['EEG']] = relationship(back_populates='experiments', secondary="association_table")
    eegs: Mapped[List['EEG']] = relationship(back_populates='experiments', secondary="result_association")
    # Metric data will be stored in dynamically created tables


class ResultAssociation(Base):
    __tablename__ = "result_association"
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiment.id"), primary_key=True)
    eeg_id: Mapped[str] = mapped_column(ForeignKey("eeg.id"), primary_key=True)
    result_path: Mapped[Optional[str]]

class Alchemist:
# functions to modify tables in the database

    @staticmethod
    def make_session(engine) -> Session:
        """
        Create a new SQLAlchemy session from the provided engine.
        
        Args:
            engine: SQLAlchemy engine to create session from
            
        Returns:
            A new SQLAlchemy session
        """
        Session = sessionmaker(engine)
        return Session()

    @staticmethod
    def remove_table(engine, table_name: str, del_from_metadata: bool = True) -> None:
        """
        Remove a table from the database and optionally from the metadata.
        
        Args:
            engine: SQLAlchemy engine connected to the database
            table_name: Name of the table to remove
            del_from_metadata: If True, also removes the table from the metadata
            
        Returns:
            None
        """
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

    @staticmethod
    def add_column(engine, table_name: str, column_name: str, column_type: str) -> None:
        """
        Add a new column to an existing table.
        
        Args:
            engine: SQLAlchemy engine connected to the database
            table_name: Name of the table to which the column will be added
            column_name: Name of the new column
            column_type: SQL type of the new column
            
        Returns:
            None
        """
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

    @staticmethod
    def add_multiple_columns(engine, table_name: str, column_names: List[str], column_types: Union[str, List[str]]):
        """
        Add multiple columns to an existing table.


        :param engine: SQLAlchemy engine connected to the database.
        :param table_name: Name of the table to which columns will be added.
        :param column_names: A list of column names (list of strings).
        :param column_types: sql type to apply to the added columns, either a single type (string) or a list of types (list of strings).
        """
        try:
            # Convert single string to list if necessary
            if isinstance(column_types, str):
                column_types = [column_types] * len(column_names)

            with Session(engine) as session:
                for column_name, type in zip(column_names, column_types):
                    # Compile the column type for the specific database dialect
                    # Execute the ALTER TABLE command
                    stmt = text(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {type}')
                    session.execute(stmt)
                    print(f"Column {column_name} added successfully.")

                session.commit()
                print(f"Columns commited successfully.")
        except SQLAlchemyError as e:
            print(f"Error: {e}")

    @staticmethod
    def remove_column(engine, table_name: str, column_name: str) -> None:
        """
        Remove a column from an existing table.
        
        Args:
            engine: SQLAlchemy engine connected to the database
            table_name: Name of the table from which the column will be removed
            column_name: Name of the column to remove
            
        Returns:
            None
        """
        try:
            # Execute the ALTER TABLE command to drop the column
            stmt = text(f'ALTER TABLE {table_name} DROP COLUMN {column_name}')
            with engine.connect() as connection:
                connection.execute(stmt)
                print(f"Column {column_name} removed successfully.")
        except SQLAlchemyError as e:
            print(f"Error: {e}")

    # function to retrieve data from the databse
    @staticmethod
    def find_entries(session: Session, table_class: type, **kwargs) -> List[Any]:
        """
        Find entries in the table based on given parameters.
    
        Args:
            session: SQLAlchemy session connected to the database
            table_class: The ORM class representing the table
            kwargs: Column-value pairs to filter the query
            
        Returns:
            List of matching ORM objects, or empty list if none found or error occurs
        """
        try:
            query = select(table_class).filter_by(**kwargs)
            result = session.scalars(query).all()
            return result
        except SQLAlchemyError as e:
            print(f"Error: {e}")
            return []

    @staticmethod
    def get_column_value_pairs(orm_object: Any) -> Dict[str, Any]:
        """
        Retrieve column-value pairs from an SQLAlchemy ORM object as a dictionary.
    
        Args:
            orm_object: The SQLAlchemy ORM object
            
        Returns:
            A dictionary containing column-value pairs
        """
        table_class = type(orm_object)
        column_value_pairs = {column.name: getattr(orm_object, column.name) for column in table_class.__table__.columns}
        return column_value_pairs

    @staticmethod
    def get_result_path_from_ids(session: Session, experiment_id: str, eeg_id: str) -> Optional[str]:
        """
        Retrieve the result path from the ResultAssociation table based on experiment_id and eeg_id.
    
        Args:
            session: SQLAlchemy session object
            experiment_id: ID of the experiment
            eeg_id: ID of the EEG
            
        Returns:
            The result path if found, None otherwise
        """
        results = Alchemist.find_entries(session, ResultAssociation, experiment_id=experiment_id, eeg_id=eeg_id)
        if results:
            return results[0].result_path
        else:
            return None

    # function to add data to the database
    @staticmethod
    def add_metric_data_table(con, experiment_id: str, eeg_id: str, df: pd.DataFrame, table_exists: str = 'append') -> Optional[str]:
        """
        Add metric data to the database for a specific experiment and EEG.
        Creates a table named 'data_experiment_{experiment_id}_eeg_{eeg_id}'.

        Parameters:
        - engine: SQLAlchemy engine or connection
        - experiment_id: ID of the experiment
        - eeg_id: ID of the EEG
        - df: DataFrame containing channel data
        - table_exists: Action to take if the table already exists ('append', 'replace')
        """
        try:
            # if con is a session, get the connection
            if isinstance(con, Session):
                con = con.connection()

            # Create table name
            table_name = f"data_experiment_{experiment_id}"

            # Add experiment_id and eeg_id as columns to the DataFrame
            df_with_ids = df.copy()
            df_with_ids['eeg_id'] = eeg_id

            # Reorder columns to have IDs first
            cols = ['eeg_id'] + [col for col in df_with_ids.columns if
                                              col not in ['experiment_id', 'eeg_id']]
            df_new = df_with_ids[cols]

            # Make sure data is appended and unique
            if table_exists == 'append':
                try:
                    df_old = pd.read_sql_table(table_name, con)
                    df_merged = pd.concat([df_old, df_new], ignore_index=True)
                    df_merged_unique = df_merged.drop_duplicates().reset_index(drop=True)
                    print(f"Table {table_name} exist, appending new rows to existing table")
                except ValueError:
                    print(f"Table {table_name} does not exist, creating new table")
                    df_merged_unique = df_new
                except Exception as e:
                    print(f"Error reading existing table: {e}")
                    return None
            elif table_exists == 'replace':
                df_merged_unique = df_new
            else:
                print(f'table_exists argument {table_exists} is not valid, must be either append or replace')
                return None

            # Add data to SQL database
            df_merged_unique.to_sql(
                name=table_name,
                con=con,
                if_exists='replace',  # 'replace' will drop and recreate the table if it exists
                index=False  # Include the index as a column
            )

            print(f"Successfully created and populated table: {table_name}")
            return table_name

        except Exception as e:
            print(f"Error creating metric data table: {e}")
            return None

    @staticmethod
    def create_unique_id(session: Session, table_class: Type[Base], max_retries=100) -> str:
        """
        Create a unique ID that doesn't exist in the specified table.
        
        Args:
            session: SQLAlchemy session connected to the database
            table_class: The ORM class representing the table (DataSet, EEG, or Experiment)
            
        Returns:
            A unique ID string that doesn't exist in the table
        """
        attempt = 0
        while attempt < max_retries:
            new_id = str(uuid.uuid4().hex)
            # Check if the ID already exists in the table
            existing = Alchemist.find_entries(session, table_class, id=new_id)
            if not existing:
                return new_id
            attempt += 1
            print(f"ID collision detected, generating new ID")
        raise RuntimeError(f"Failed to generate a unique ID after {max_retries} attempts.")

    @staticmethod
    def add_or_update_eeg_entry(session: Session, dataset_id: str, filepath: str, 
                                filename: str, file_extension: str) -> Optional[EEG]:
        """
        Initialize or retrieve an EEG entry in the database.
    
        Parameters:
        - sqlpath: Path to the SQLite database
        - dataset_id: ID of the dataset this EEG belongs to
        - filepath: Path to the EEG file
        - filename: Name of the EEG file (without extension)
        - file_extension: Extension of the EEG file
    
        Returns:
        - eeg_id: The id of the EEG object that was created or retrieved
        """
        # Check if the EEG already exists in the database
        matching_eegs = Alchemist.find_entries(session, EEG,
                                    dataset_id=dataset_id,
                                    filename=filename,
                                    filepath=filepath,
                                    filetype=file_extension)
        if len(matching_eegs) == 0:
            unique_id = Alchemist.create_unique_id(session, EEG)
            eeg = EEG(id=unique_id,
                     filename=filename,
                     dataset_id=dataset_id,
                     filepath=filepath,
                     filetype=file_extension)
            session.add(eeg)
            print(f"Created new EEG entry: {filename}")
        elif len(matching_eegs) == 1:
            print(f"Found matching EEG in the dataset: {filename}")
            eeg = matching_eegs[0]
        else:
            print(f"Multiple EEGs in the dataset that match {filename}, please manually check")
            return None
        session.commit()
        return eeg

    @staticmethod
    def add_or_update_experiment(session: Session, metric_set_name: str, run_name: str, fs: Optional[int] = None,
                             start: Optional[int] = None, stop: Optional[int] = None, 
                             lower_cutoff: Optional[float] = None, upper_cutoff: Optional[float] = None, 
                             window_len: Optional[int] = None, window_overlap: Optional[int] = None, 
                             montage: Optional[str] = None) -> Experiment:
        """
        Initialize or retrieve a MetricSet entry in the database.

        Parameters:
        - sqlpath: Path to the SQLite database
        - eeg_id: ID of the EEG this metric set belongs to
        - metric_set_name: Name of the metric set
        - signal_len: Length of the signal in samples
        - fs: Sampling frequency
        - lfreq: Lower cutoff frequency for filtering
        - hfreq: Upper cutoff frequency for filtering
        - ep_dur: Duration of each epoch in seconds
        - ep_overlap: Overlap between epochs in seconds
        - montage: Montage used for the EEG

        Returns:
        - MetricSet: The MetricSet object that was created or retrieved
        """
        # Check if metric set already exists for this EEG
        matching_experiments = Alchemist.find_entries(
            session,
            Experiment,
            metric_set_name=metric_set_name,
            run_name=run_name,
        )

        if len(matching_experiments) == 0:
            unique_id = Alchemist.create_unique_id(session, Experiment)
            experiment = Experiment(
                id=unique_id,
                metric_set_name=metric_set_name,
                run_name=run_name,
                fs=fs,
                start=start,
                stop=stop,
                window_len=window_len,
                window_overlap=window_overlap,
                lower_cutoff=lower_cutoff,
                upper_cutoff=upper_cutoff,
                montage=montage
            )
            session.add(experiment)
            session.commit()
            print(f"Created new metric set: {metric_set_name}")
            return experiment
        elif len(matching_experiments) == 1:
            print(f"Found existing metric set: {metric_set_name}")
            return matching_experiments[0]
        else:
            raise ValueError(f"Multiple metric sets found for {metric_set_name}")

    @staticmethod
    def add_or_update_dataset(session: Session, dataset_name: str, dataset_path: str, 
                             dataset_description: str) -> Optional[DataSet]:
        """
        Add or update a dataset in the database.

        Parameters:
        - sqlpath: Path to the SQLite database
        - dataset_name: Name of the dataset
        - dataset_path: Path to the dataset
        - dataset_description: Description of the dataset

        Returns:
        - DataSet: The dataset object that was created or retrieved
        """
        # Add a dataset to the sqlite database
        # Check if the dataset already exists in our database
        matching_datasets = Alchemist.find_entries(session, DataSet, name=dataset_name, path=dataset_path)
        if len(matching_datasets) == 0:
            unique_id = Alchemist.create_unique_id(session, DataSet)
            dataset = DataSet(id=unique_id, name=dataset_name, path=dataset_path, description=dataset_description)
            session.add(dataset)
            print(f"Created new dataset: {dataset_name}")
        elif len(matching_datasets) == 1:
            print(f"Found matching dataset in database, updating description if necessary")
            dataset = matching_datasets[0]
            dataset.description = dataset_description
        else:
            print('Multiple datasets in the database that match name and path, please manually check')
            return None
        session.commit()
        return dataset

    @staticmethod
    def add_result_path(session: Session, experiment_id: str, eeg_id: str, 
                       result_path: str) -> Optional[ResultAssociation]:
        matching_results = Alchemist.find_entries(session, ResultAssociation, experiment_id=experiment_id, eeg_id=eeg_id)
        if len(matching_results) == 0:
            print('No result found for experiment and eeg, please ensure experiment and eeg are in the database')
            return None
        elif len(matching_results) == 1:
            matching_results[0].result_path = result_path
            session.commit()
            return matching_results[0]
        else:
            print('Multiple results in the database that match experiment and eeg, please manually check')
            return None

    @staticmethod
    def initialize_tables(path: Optional[str] = None, path_is_relative: bool = True):
        if path:
            if path_is_relative:
                engine = create_engine(f"sqlite+pysqlite:///{path}")
            else:
                engine = create_engine(f"sqlite+pysqlite:////{path}")
        else:
            engine = create_engine("sqlite+pysqlite://:memory:")
        Base.metadata.create_all(bind=engine)
        return engine

    # functions to test functionality

    @staticmethod
    def test_adding_data(db_path: str) -> None:
        """Test the data addition pipeline with a sample EEG dataset and metrics."""
        # Define the metric name first (needed for metric set creation)
        engine = Alchemist.initialize_tables(db_path)

        with Session(engine) as session:
            # 1. Create a test dataset
            dataset = Alchemist.add_or_update_dataset(
                session,
                dataset_name='EEG Study Dataset',
                dataset_path='/data/eeg_study',
                dataset_description='Dataset for EEG study on cognitive functions'
            )
            # 2. Create a test EEG entry
            eeg = Alchemist.add_or_update_eeg_entry(
                session,
                dataset_id=dataset.id,
                filepath='/data/eeg_study',
                filename='subject_01_session_01',
                file_extension='.edf'
            )
            # 3. Create a test metric set
            result_path = '/results/alpha_power'
            experiment = Alchemist.add_or_update_experiment(
                session,
                metric_set_name='Alpha power',
                run_name='low_freq',
                fs=250,          # Sample frequency
                lower_cutoff=8,         # Alpha band lower cutoff
                upper_cutoff=13,        # Alpha band upper cutoff
                window_len=4,        # 4-second epochs
                window_overlap=2     # 2-second overlap
            )
            if not eeg in experiment.eegs:
                experiment.eegs.append(eeg)

            Alchemist.add_result_path(session, experiment.id, eeg.id, result_path)

            # 4. Create sample channel data as a DataFrame
            channel_data = pd.DataFrame({
                'Fp1': [0.75, 1],
                'Fp2': [0.82, 2.5],
                'F3': [0.65, 4.5],
                'F4': [0.71, 6]
            })

            # 5. Add the channel data to the database
            Alchemist.add_metric_data_table(
                con=session.connection(),
                experiment_id=experiment.id,
                eeg_id=eeg.id,
                df=channel_data
            )
            session.commit()
            print("Test data added successfully.")

if __name__ == "__main__":
    # Run the test functions
    print("Running database tests...")
    db_path = 'test.sqlite'
    Alchemist.test_adding_data(db_path)
    print("All tests completed successfully.")