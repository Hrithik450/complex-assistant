from sqlalchemy import Column, String, Text, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Thread(Base):
    __tablename__ = "thread"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    role = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    embedding_vector = Column(Vector(3072))