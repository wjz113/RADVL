"""RADVL recorder diagnostics."""
import json
from pathlib import Path
import shutil

from radvl.recorder_balanced import RADVLResidualBalancedRecorder
from radvl.recorder_base import append_csv
from radvl.data_protocol import file_hash
from radvl.gradient_diagnostics import AGGREGATE_LABEL, AGGREGATE_SOURCES, GROUP_NAMES, LATENT_GEOMETRY_SIDES, PARAMETER_GROUPS, TERM_KEYS, audit_interval, parameter_groups


AUDIT_SOURCE_FILES = ('gradient_diagnostics.py',
                      'recorder_diagnostics.py',
                      'training_audit.py',
                      'test_gradient_diagnostics.py',
                      'objectives_balanced.py')


class RADVLResidualAuditRecorder(RADVLResidualBalancedRecorder):
    def __init__(self, directory, args, k, dataset, protocol, requested_args):
        super().__init__(directory, args, k, dataset, protocol, requested_args)
        root = Path(__file__).resolve().parent
        config = json.loads((self.directory / 'run_config.json').read_text(encoding='utf-8'))
        for name in AUDIT_SOURCE_FILES:
            shutil.copy2(root / name, self.directory / 'source' / name)
            config['input_and_source_sha256']['source/' + name] = file_hash(root / name)
        config.update(
            model='radvl_aligned_covariance_CoVAE_weighted_graphs_residual_prediction_balanced_audited',
            gradient_audit=dict(
                enabled=bool(audit_interval()),
                every_epochs=audit_interval(),
                environment_variable='GRADIENT_AUDIT_EVERY (0 disables the audit)',
                audited_terms={label: key for label, key in TERM_KEYS},
                aggregate=dict(label=AGGREGATE_LABEL, sources=list(AGGREGATE_SOURCES),
                               how='gradient of the sum, obtained by adding the sub-term gradients (exact by '
                                   'linearity, no extra backward pass); excluded from the share normalisation '
                                   'because it would be counted three times'),
                weights='each sub-term is already multiplied by its actual weight: reconstruction and KL by '
                        'lambda_av, the covariance penalty by lambda_av * lambda_cov, the alignment term by '
                        'lambda_align',
                measured=['per-term gradient norms over all parameters, over the parameters shared by at least '
                          'two independent terms, and per parameter group',
                          'ratio_to_prediction and share_of_total_gradient',
                          'pairwise cosines on the parameters both terms touch',
                          'per-side latent covariance, correlation and effective rank from the posterior means',
                          'the covariance penalty recomputed from the latents next to the objective value'],
                parameter_groups={name: list(prefixes) for name, prefixes in PARAMETER_GROUPS},
                implementation='torch.autograd.grad(..., retain_graph=True) inside the objective, before the '
                               'training loop calls backward; geometry under torch.no_grad() on detached float64 '
                               'CPU copies',
                observer_only='autograd.grad writes no .grad, consumes no RNG and frees no graph, the geometry '
                              'runs under no_grad on detached copies, so the audited run is bit-identical to the '
                              'same run without the audit. This is asserted by test_gradient_diagnostics.py, and the CPU '
                              'and initialised CUDA RNG states are fingerprinted before and after every audit and '
                              'reported in the rng_state_unchanged columns of gradient_audit.csv',
                cost='five extra graph traversals per audited epoch plus one eigendecomposition per side',
                deliberately_unchanged='the covariance formula, the model structure, the adaptive A3 alignment '
                                       'temperature, the frozen protocol and the training loop'),
            evidence_notes=[
                'Loss values do not measure gradient contribution. In the released logs the auxiliary objectives '
                'were 0.75% of the summed total loss; the audit measures what they actually contribute to the '
                'gradient, which is what the review asks for.',
                'Measured before this extension, with lambda_cov=10 against lambda_cov=0 and everything else '
                'identical: test AUC 0.890101 against 0.890087 (per-fold differences +0.0019, +0.00001, -0.0021, '
                '+0.0003, -0.0001, cancelling), so the covariance penalty did not change the ranking, and the '
                'association-term gradient norm was 0.398 against 0.415 with identical extremes. This run measures '
                'the covariance contribution directly instead of inferring it from that comparison.',
                'Measured latent geometry with lambda_cov=10 against 0: in the one fold that ran 207 epochs the mean '
                'absolute off-diagonal correlation fell from 0.3548 to 0.2455 and the effective rank rose from 4.7 '
                'to 8.4 out of 123; the other four folds (88-98 epochs, early stopped) were equal to three decimals. '
                'The covariance penalty is therefore binding on its own quantity, but only given enough steps.'],
            limitations=list(config.get('limitations', []))
                        + ['The audit measures local gradients at the sampled epoch, not the effect of removing a '
                           'term; absence of conflict is not proof of usefulness.',
                           'gradient_audit.csv changed schema with this extension: runs produced before it have a '
                           'single association term and no geometry columns.',
                           'This entry has not been trained by the editing task that created it.'])
        config['notes'].append('Gradient audit extended to the CoVAE sub-terms and the latent geometry; see '
                               'gradient_audit.csv, gradient_cosine.csv and latent_geometry.csv.')
        self.write_json('run_config.json', config)

    def epoch(self, fold, epoch, total, losses, model, optimizer, compute_seconds):
        super().epoch(fold, epoch, total, losses, model, optimizer, compute_seconds)
        audit = getattr(model, 'gradient_audit', None)
        if not audit:
            return
        prefix = dict(fold=fold, epoch=epoch, call_index=audit['call_index'],
                      parameter_count=audit['parameter_count'],
                      unassigned_parameter_count=audit['unassigned_parameter_count'],
                      rng_state_unchanged=audit['rng_state_unchanged'],
                      rng_state_unchanged_all=audit['rng_state_unchanged_all'])
        # update() overwrites duplicate keys instead of raising, so a field added to the term
        # rows can never collide with a field added here.
        rows = []
        for row in audit['terms']:
            merged = dict(prefix)
            merged.update(row)
            rows.append(merged)
        append_csv(self.directory / 'gradient_audit.csv', rows)
        rows = []
        for row in audit['cosines']:
            merged = dict(fold=fold, epoch=epoch)
            merged.update(row)
            rows.append(merged)
        append_csv(self.directory / 'gradient_cosine.csv', rows)
        rows = []
        for side, _ in LATENT_GEOMETRY_SIDES:
            merged = dict(fold=fold, epoch=epoch, call_index=audit['call_index'], side=side)
            merged.update(audit['geometry'][side])
            rows.append(merged)
        append_csv(self.directory / 'latent_geometry.csv', rows)
        groups = parameter_groups(model)
        append_csv(self.directory / 'gradient_audit_coverage.csv', [dict(
            fold=fold, epoch=epoch, group=group,
            is_named_group=group != 'other',
            parameter_tensors=len(groups[group]),
            parameter_elements=sum(int(parameter.numel()) for _, parameter in groups[group]))
            for group in GROUP_NAMES])
