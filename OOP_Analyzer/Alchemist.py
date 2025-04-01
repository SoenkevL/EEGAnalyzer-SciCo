from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import MetaData, ForeignKey, Table, Column, Integer, String, create_engine
from typing import List, Optional

# declaring a shorthand for the declarative base class
class Base(DeclarativeBase):
    pass

# defining the classes for our project with the correct meta data
# we will first go for a simple approach leaving the actual calculations in csv format as is the case now
class DataSet(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    path_from_root: Mapped[str] = mapped_column(String, nullable=False)
    path_from_project: Mapped[Optional[str]]
    description: Mapped[Optional[str]]

    eegs: Mapped[List["EEG"]] = relationship(back_populates="dataset_id")


class EEG(Base):
    __tablename__ = "EEG"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id = mapped_column(ForeignKey("dataset.id"))
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filetype: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)#
    description: Mapped[Optional[str]]

    dataset: Mapped[DataSet] = relationship(back_populates='eegs')
    metrics: Mapped[List['Metric']] = relationship(back_populates='eeg')


class Metric(Base):
    __tablename__ = "metric"

    id: Mapped[int] = mapped_column(primary_key=True)
    eeg_id = mapped_column(ForeignKey("EEG.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)

    eeg: Mapped[EEG] = relationship(back_populates='metrics')
    processing_params: Mapped[List['MetricParameters']] = relationship(back_populates='parent_metric')


class MetricParameters(Base):
    __tablename__ = "metric_parameter"

    id: Mapped[int] = mapped_column(primary_key=True)
    metric_id = mapped_column(ForeignKey("metric.id"), nullable=False)
    description: Mapped[Optional[str]]
    signal_len: Mapped[int]
    fs: Mapped[Optional[int]]
    window_len: Mapped[Optional[int]]
    window_overlap: Mapped[Optional[int]]
    lower_cutoff: Mapped[Optional[float]]
    upper_cutoff: Mapped[Optional[float]]
    montage: Mapped[Optional[str]]

    results: Mapped[List['MetricResults']] = relationship(back_populates="metric_parameter")


class MetricResults(Base):
    __tablename__ = "metric_result"

    id: Mapped[int] = mapped_column(primary_key=True)
    parameter_id = mapped_column(ForeignKey("metric_parameter.id"), nullable=False)
    result_path: Mapped[str] = mapped_column(String, nullable=False)
    start_sample: Mapped[int]
    end_sample: Mapped[int]

    metric_parameter: Mapped[MetricParameters] = relationship(back_populates='results')
    results_data: Mapped[List['MetricResultsData']] = relationship(back_populates='metric_result')

class MetricResultsData(Base):
    def __init__(self, table_name: str):
        super().__init__()  # Properly initialize the superclass
        self.__tablename__ = table_name  # Set the table name

    id: Mapped[int] = mapped_column(primary_key=True)
    result_id = mapped_column(ForeignKey("metric_result.id"), nullable=False)
    metric_result: Mapped[MetricResults] = relationship(back_populates='results_data')


def initialize_tables(path=None, path_is_relative=True):
    if path:
        if path_is_relative:
            engine = create_engine(f"sqlite+pysqlite:///{path}")
        else:
            engine = create_engine(f"sqlite+pysqlite:////{path}")
    else:
        engine = create_engine("sqlite+pysqlite://:memory:")
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    db_path = 'test.sqlite'
    initialize_tables(db_path)