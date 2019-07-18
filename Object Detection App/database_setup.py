import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()


class Shop(Base):
    __tablename__ = 'shop'
    id=Column(Integer,primary_key=True)
    name=Column(String(30),nullable=False)
    desc=Column(String(300),nullable=False)
    price=Column(Integer,nullable=False)



engine = create_engine('sqlite:///myshop.db')


Base.metadata.create_all(engine)
