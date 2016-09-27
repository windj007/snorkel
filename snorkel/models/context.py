from .meta import SnorkelBase, snorkel_postgres
from sqlalchemy import Column, String, Integer, Text, ForeignKey
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import relationship, backref
from sqlalchemy.types import PickleType

class Context(SnorkelBase):
    """A piece of content."""
    __tablename__ = 'context'
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'context',
        'polymorphic_on': type
    }


class Corpus(Context):
    """
    A Corpus holds a set of Documents.

    Default iterator is over (Document, Sentence) tuples.
    """
    __tablename__ = 'corpus'
    id = Column(Integer, ForeignKey('context.id'), unique=True, nullable=False)
    name = Column(String, primary_key=True)

    __mapper_args__ = {
        'polymorphic_identity': 'corpus',
    }

    def __repr__(self):
        return "Corpus (" + str(self.name) + ")"

    def __iter__(self):
        """Default iterator is over (document, document.sentences) tuples"""
        for doc in self.documents:
            yield (doc, doc.sentences)


    def get_sentences(self):
        return [sentence for doc in self.documents for sentence in doc.sentences]

    def get_tables(self):
        return [table for doc in self.documents for table in doc.tables]

    def get_phrases(self):
        return [phrase for doc in self.documents for phrase in doc.phrases]


class Document(Context):
    """An object in a Corpus."""
    __tablename__ = 'document'
    id = Column(Integer, ForeignKey('context.id'), unique=True, nullable=False)
    name = Column(String, primary_key=True)
    corpus_id = Column(Integer, ForeignKey('corpus.id'), primary_key=True)
    corpus = relationship('Corpus', backref=backref('documents', cascade='all, delete-orphan'), foreign_keys=corpus_id)
    file = Column(String)
    attribs = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'document',
    }

    def __repr__(self):
        return "Document" + str((self.name, self.corpus))


class Sentence(Context):
    """A sentence Context in a Document."""
    __tablename__ = 'sentence'
    id = Column(Integer, ForeignKey('context.id'), unique=True)
    document_id = Column(Integer, ForeignKey('document.id'), primary_key=True)
    document = relationship('Document', backref=backref('sentences', cascade='all, delete-orphan'), foreign_keys=document_id)
    position = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    xmltree = Column(PickleType)
    if snorkel_postgres:
        words = Column(postgresql.ARRAY(String), nullable=False)
        char_offsets = Column(postgresql.ARRAY(Integer), nullable=False)
        lemmas = Column(postgresql.ARRAY(String))
        poses = Column(postgresql.ARRAY(String))
        dep_parents = Column(postgresql.ARRAY(Integer))
        dep_labels = Column(postgresql.ARRAY(String))
    else:
        words = Column(PickleType, nullable=False)
        char_offsets = Column(PickleType, nullable=False)
        lemmas = Column(PickleType)
        poses = Column(PickleType)
        dep_parents = Column(PickleType)
        dep_labels = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'sentence',
    }

    def __repr__(self):
        return "Sentence" + str((self.document, self.position, self.text))


class Table(Context):
    __tablename__ = 'table'
    id = Column(Integer, ForeignKey('context.id'))
    document_id = Column(Integer, ForeignKey('document.id'), primary_key=True)
    position = Column(Integer, primary_key=True)

    document = relationship('Document', backref=backref('tables', cascade='all, delete-orphan'), foreign_keys=document_id)

    text = Column(Text, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'table',
    }

    def __repr__(self):
        return "Table" + str((self.document.name, self.position))


class Cell(Context):
    __tablename__ = 'cell'
    id = Column(Integer, ForeignKey('context.id'))
    document_id = Column(Integer, ForeignKey('document.id'), primary_key=True)
    table_id = Column(Integer, ForeignKey('table.id'), primary_key=True)
    position = Column(Integer, primary_key=True)

    document = relationship('Document', backref=backref('cells', cascade='all, delete-orphan'), foreign_keys=document_id)
    table = relationship('Table', backref=backref('cells', cascade='all, delete-orphan'), foreign_keys=table_id)

    text = Column(Text, nullable=False)
    row_num = Column(Integer)
    col_num = Column(Integer)
    html_tag = Column(Text)
    if snorkel_postgres:
        html_attrs = Column(postgresql.ARRAY(String))
        html_anc_tags = Column(postgresql.ARRAY(String))
        html_anc_attrs = Column(postgresql.ARRAY(String))
    else:
        html_attrs = Column(PickleType)
        html_anc_tags = Column(PickleType)
        html_anc_attrs = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'cell',
    }

    def __repr__(self):
        return "Cell" + str((self.document.name, self.table.position, self.position, self.text))

    def spans(self, axis):
        """Returns true if cell spans entire axis"""
        assert axis in ('row', 'col')
        axis_name = axis + '_num'
        axis_cells = [c for c in self.table.cells if getattr(c, axis_name)==getattr(self, axis_name)]
        return True if len(axis_cells) == 1 and axis_cells[0] == self else False

    def head_cell(self, axis, induced=False):
        """Return first aligned cell along given axis

        If we are already at the head cell for this axis, return None

        The first aligned cell along an axis is the one whose other axis is 0
        """
        if axis not in ('row', 'col'): raise ValueError("Axis must equal 'row' or 'col'")

        other_axis = 'col' if axis == 'row' else 'row'
        other_axis_name = other_axis + '_num'
        if getattr(self, other_axis_name) == 0: return None
        cells = [cell for cell in self.aligned_cells(axis=axis) 
                 if getattr(cell, other_axis_name) == 0]
        if not cells: return None

        assert len(cells) == 1

        head_cell = cells[0]
        
        if induced and not head_cell.text.isspace():
            return head_cell.first_aligned_nonempty_cell(other_axis)

        return head_cell

    def aligned_cells(self, axis, induced=False):
        """Return list of aligned cells along given axis"""

        if axis not in ('row', 'col'): raise ValueError("Axis must equal 'row' or 'col'")

        axis_name = axis + '_num'
        cells = [ cell for cell in self.table.cells
                  if getattr(cell,axis_name) == getattr(self,axis_name)
                  and cell != self ]

        if induced:
            other_axis = 'col' if axis == 'row' else 'row'
            def induced_or_real(c):
                return c if c.text and not c.text.isspace() else \
                       c.first_aligned_nonempty_cell(other_axis)
            cells = [induced_or_real(cell) for cell in cells
                     if induced_or_real(cell) is not None]
        return cells

    def first_aligned_nonempty_cell(self, axis, dir='up'):
        """Return first non-empty cell along axis in given direction

        Currently, 'dir' must be 'up'

        If no such cell exists, None is returned. If this is the first cell,
        None is returned.

        Axis is 'row' or 'col'. Dir is 'up' (decreasing) or 'down' (increasing).
        """
        if dir != 'up': raise NotImplementedError("Please use dir='up' for now")
        if axis not in ('row', 'col'): raise ValueError("Axis must equal 'row' or 'col'")
        
        axis_name = axis + '_num'
        other_axis = 'col' if axis == 'row' else 'row'
        other_axis_name = other_axis + '_num'
        # get cells aligned to self that appear before self and that aren't empty
        aligned_cells = [cell for cell in self.aligned_cells(axis)
                         if getattr(cell,other_axis_name) < getattr(self,other_axis_name)
                         and cell.text and not cell.text.isspace()]
        # pick the last cell among the ones identified above
        aligned_cells = sorted(aligned_cells, key=lambda x: getattr(x,other_axis_name), reverse=True)
        if aligned_cells:
            out_cell = aligned_cells[0]
        else:
            out_cell = None

        return out_cell

class Phrase(Context):
    __tablename__ = 'phrase'
    id = Column(Integer, ForeignKey('context.id'))
    document_id = Column(Integer, ForeignKey('document.id'), primary_key=True)
    table_id = Column(Integer, ForeignKey('table.id'), primary_key=True)
    cell_id = Column(Integer, ForeignKey('cell.id'), primary_key=True)
    position = Column(Integer, primary_key=True)

    document = relationship('Document', backref=backref('phrases', cascade='all, delete-orphan'), foreign_keys=document_id)
    table = relationship('Table', backref=backref('phrases', cascade='all, delete-orphan'), foreign_keys=table_id)
    cell = relationship('Cell', backref=backref('phrases', cascade='all, delete-orphan'), foreign_keys=cell_id)

    text = Column(Text, nullable=False)
    xmltree = Column(PickleType)
    row_num = Column(Integer)
    col_num = Column(Integer)
    html_tag = Column(Text)
    if snorkel_postgres:
        html_attrs = Column(postgresql.ARRAY(String))
        html_anc_tags = Column(postgresql.ARRAY(String))
        html_anc_attrs = Column(postgresql.ARRAY(String))
        words = Column(postgresql.ARRAY(String), nullable=False)
        char_offsets = Column(postgresql.ARRAY(Integer), nullable=False)
        lemmas = Column(postgresql.ARRAY(String))
        poses = Column(postgresql.ARRAY(String))
        dep_parents = Column(postgresql.ARRAY(Integer))
        dep_labels = Column(postgresql.ARRAY(String))
    else:
        html_attrs = Column(PickleType)
        html_anc_tags = Column(PickleType)
        html_anc_attrs = Column(PickleType)
        words = Column(PickleType, nullable=False)
        char_offsets = Column(PickleType, nullable=False)
        lemmas = Column(PickleType)
        poses = Column(PickleType)
        dep_parents = Column(PickleType)
        dep_labels = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'phrase',
    }

    def __repr__(self):
        if self.table is not None and self.cell is not None:
            return "Phrase" + str((self.document.name, self.table.position, self.cell.position, self.position, self.text))
        else:
            return "Phrase" + str((self.document.name, None, None, self.position, self.text))
            
