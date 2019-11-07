from application.extensions import db


class Corpus(db.Model):
    """Data model for a corpus."""

    __tablename__ = 'corpus'
    id = db.Column(db.Integer,
                   primary_key=True)
    name = db.Column(db.String(255),
                     nullable=False)
    status = db.Column(db.String(32),
                       index=False,
                       nullable=False)

    def __repr__(self):
        return '<Corpus {}>'.format(self.name)


class CorpusResult(db.Model):
    """Data model for result returned from Watson NLU service."""

    __tablename___ = 'corpus_result'
    id = db.Column(db.Integer, primary_key=True)
    corpus_id = db.Column(db.Integer,
                          db.ForeignKey("corpus.id"),
                          nullable=False)
    name = db.Column(db.String(255), nullable=False)
    data = db.Column(db.BINARY, nullable=True)

    def __repr__(self):
        return '<Corpus Result {}>'.format(self.id)
